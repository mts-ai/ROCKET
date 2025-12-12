import torch
from tqdm import tqdm
from transformers.utils import logging
import os
import pickle

logger = logging.get_logger(__name__)


class Hook:
    def __init__(self, module, backward=False):
        self.calib = None
        self.calib_out = None
        self.total_tokens = 0
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            inp = input[0].detach()
            out = output.detach()
            # Optional: prevent explosion
            if (inp.isnan().any() or inp.isinf().any()):
                return
            if (out.isnan().any() or out.isinf().any()):
                return

            if inp.dtype != torch.float32:
                inp = inp.float()
            if out.dtype != torch.float32:
                out = out.float()

            # Clamp extreme values (tune threshold based on model)
            inp = torch.clamp(inp, -10.0, 10.0)
            out = torch.clamp(out, -10.0, 10.0)

            B, S, D = inp.shape
            if len(out.shape) != len(inp.shape):
                out = out.transpose(-1, 0)
                out = out.reshape(B, S, -1)
            B2, S2, D2 = out.shape
            inp_flat = inp.view(-1, D)  # [B*S, D]
            out_flat = out.view(-1, D2)
            # Use double for numerical stability
            local_g = (inp_flat.t() @ inp_flat).double()  # [D, D]
            local_g_out = (out_flat.t() @ out_flat).double()  # [D2, D2]

            if self.calib is None:
                self.calib = local_g.cpu()
            else:
                self.calib += local_g.cpu()

            if self.calib_out is None:
                self.calib_out = local_g_out.cpu()
            else:
                self.calib_out += local_g_out.cpu()

            self.total_tokens += B * S

    def get_covariance(self):
        """Return average Gram matrix."""
        if self.calib is None:
            return None
        return (self.calib / self.total_tokens).float()

    def get_covariance_out(self):
        """Return average Gram matrix."""
        if self.calib_out is None:
            return None
        return (self.calib_out / self.total_tokens).float()

    def close(self):
        self.hook.remove()
        del self.calib
        del self.total_tokens

class Calib:
    @staticmethod
    def save(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_calib_data(group, name, save_path=None, out=False):
        assert save_path is not None
        # calibration data for up and gate is the same
        if name == 'mlp.gate_proj':
            name = 'mlp.up_proj'
        if name == "self_attn.q_proj" or name == "self_attn.v_proj":
            name = "self_attn.k_proj"
        data = None
        file_path = os.path.join(save_path, name)
        for item in group:
            file_name = os.path.join(file_path, "{}.pkl".format(item))
            if out:
                file_name = os.path.join(file_path, "out{}.pkl".format(item))
            if os.path.exists(file_name):
                tmp_data = Calib.load(file_name)
                if data is None:
                    data = tmp_data
                else:
                    data += tmp_data
            else:
                raise FileNotFoundError(
                    "{} not found. You should run build_calibration_dataset first!".format(file_name))
        return data
   
    @staticmethod
    def get_s_inv_s(group, name, model_type, calib_path=None):
        data = Calib.get_calib_data(group, name, calib_path).double()
        # The following code is from https://github.com/AIoT-MLSys-Lab/SVD-LLM
        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(data)
            data += (- eigenvalues[0] + 7e-6) * torch.eye(data.shape[0]).to(data.device)
            scaling_diag_matrix = torch.linalg.cholesky(data).T
            eigenvalues = None
            del eigenvalues
        invs = torch.linalg.inv(scaling_diag_matrix)
        return scaling_diag_matrix, invs


    @staticmethod
    def get_s_inv_s_robust_cholesky(group, name, model_type, calib_path=None, eps=1e-6):
        A = Calib.get_calib_data(group, name, calib_path).double()
        A = (A + A.T) / 2.0
    
        # Ensure positive definiteness
        min_eig = torch.linalg.eigvalsh(A)[0]
        if min_eig < eps:
            A = A + (eps - min_eig) * torch.eye(A.shape[0], device=A.device)
    
        L = torch.linalg.cholesky(A)
        ss = L.T
        inv_s = torch.linalg.inv(ss)
        return ss, inv_s
    @staticmethod
    def get_s_inv_s_out(group, name, model_type, calib_path=None):
        data = Calib.get_calib_data(group, name, calib_path, out=True).double()
        # The following code is from https://github.com/AIoT-MLSys-Lab/SVD-LLM
        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(data)
            data += (- eigenvalues[0] + 7e-6) * torch.eye(data.shape[0]).to(data.device)
            scaling_diag_matrix = torch.linalg.cholesky(data).T
            eigenvalues = None
            del eigenvalues
        invs = torch.linalg.inv(scaling_diag_matrix)
        return scaling_diag_matrix, invs

    @staticmethod
    def build_calibration_dataset(model, dataloader, names, model_type, save_path):
        print("Start building calibration data.")

        if model_type == "gpt2":
            tmp_model = model.transformer.h
        elif model_type == "llama2" or model_type == "llama3":
            tmp_model = model.model.layers
        elif model_type == "opt":
            tmp_model = model.model.decoder.layers
        elif model_type == "mistral":
            tmp_model = model.model.layers
        elif model_type == "clip":
            tmp_model = model.vision_model.encoder.layers
        else:
            raise NotImplementedError

        hooks = {}
        for name in names:
            hooks[name] = []
            for layer in tmp_model:
                target = layer.get_submodule(name)
                hooks[name].append(Hook(target, backward=False))
        # print(hooks)

        model.config.use_cache = False
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                if model_type == "clip":
                    inputs = {k: v.to(model.device).squeeze(1) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                else:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    out = model(**batch)
                

        assert save_path is not None
        for name in names:
            tmp_save_path = os.path.join(save_path, name)
            if not os.path.exists(tmp_save_path):
                os.makedirs(tmp_save_path)
            for i, hook in enumerate(hooks[name]):
                data = hook.calib.cpu()
                data2 = hook.calib_out.cpu()
                tmp_name = str(i) + ".pkl"
                Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                Calib.save(os.path.join(tmp_save_path, "out" + tmp_name), data2)
                hook.close()

    @staticmethod
    def build_update_dataset(model, dataloader, names, model_type, save_path):
        print("Start building update dataset.")
        if model_type == "gpt2":
            tmp_model = model.transformer
            num_layers = len(tmp_model.h)
        elif model_type == "llama2" or model_type == "mistral" or model_type == "llama3":
            tmp_model = model.model
            num_layers = len(tmp_model.layers)
        elif model_type == "opt":
            tmp_model = model.model.decoder
            num_layers = len(tmp_model.layers)
        elif model_type == "clip":
            tmp_model = model.vision_model.encoder
            num_layers = len(tmp_model.layers)
        else:
            raise NotImplementedError

        hooks = {}
        for name in names:
            hooks[name] = []
            for i in range(num_layers):
                target = tmp_model.get_submodule(name)[str(i)]
                hooks[name].append(Hook(target, backward=False))

        model.config.use_cache = False
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)

        assert save_path is not None
        for name in names:
            tmp_save_path = os.path.join(save_path, name)
            if not os.path.exists(tmp_save_path):
                os.makedirs(tmp_save_path)
            for i, hook in enumerate(hooks[name]):
                data = hook.calib.cpu()
                data2 = hook.calib_out.cpu()
                tmp_name = str(i) + ".pkl"
                Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                Calib.save(os.path.join(tmp_save_path, "out" + tmp_name), data2)
                hook.close()
