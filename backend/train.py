"""Fine-tune LoFTR on synthetic pairs. Skeleton: plug in the LoFTR loss."""
import glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import kornia.feature as KF
from pipeline import DEVICE


class SynthDataset(Dataset):
    def __init__(self, d): self.f = sorted(glob.glob(f"{d}/*.npz"))
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        z = np.load(self.f[i])
        return {"image0": torch.from_numpy(z["img0"])[None], "image1": torch.from_numpy(z["img1"])[None],
                "H": torch.from_numpy(z["H"]).float()}


def loftr_loss(out, batch):
    # TODO: use the coarse focal loss + fine L2 loss from the official LoFTR repo (zju3dv/LoFTR),
    # building GT coarse matches from batch["H"] (exact, since the DEM is fixed).
    raise NotImplementedError


def run(data_dir, epochs, state, out="models/loftr-lunar-v1.pt"):
    model = KF.LoFTR(pretrained="outdoor").to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dl = DataLoader(SynthDataset(data_dir), batch_size=2, shuffle=True)
    state.update(status="running", epoch=0, epochs=epochs)
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = loftr_loss(model(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            state["loss"] = float(loss)
        state["epoch"] = ep + 1
    torch.save(model.state_dict(), out)
    state["status"] = "done"


def export_onnx(weights="models/loftr-lunar-v1.pt", out="models/loftr-lunar-v1.onnx"):
    m = KF.LoFTR(pretrained=None); m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval()
    # NOTE: LoFTR's dict I/O may need a thin wrapper module before torch.onnx.export works.
