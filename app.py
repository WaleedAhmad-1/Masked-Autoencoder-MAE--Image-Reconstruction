import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms

device = "cpu"

# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):

        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        x = x.flatten(2).transpose(1,2)

        return x


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):

    def __init__(self, embed_dim=768, depth=6, num_heads=8):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):

        return self.encoder(x)


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):

    def __init__(self, embed_dim=384, depth=4, num_heads=8):

        super().__init__()

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=depth
        )

        self.head = nn.Linear(embed_dim, 16*16*3)

    def forward(self, x):

        x = self.decoder(x)

        x = self.head(x)

        return x


# -----------------------------
# MAE Model
# -----------------------------
class MAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch = PatchEmbed()

        self.pos_embed = nn.Parameter(torch.zeros(1,196,768))

        self.encoder = Encoder()

        self.enc_to_dec = nn.Linear(768,384)

        self.mask_token = nn.Parameter(torch.zeros(1,1,384))

        self.decoder = Decoder()

    def random_mask(self, x, mask_ratio=0.75):

        N,L,D = x.shape

        len_keep = int(L*(1-mask_ratio))

        noise = torch.rand(N,L)

        ids_shuffle = torch.argsort(noise, dim=1)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:,:len_keep]

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1,1,D)
        )

        mask = torch.ones(N,L)

        mask[:,:len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):

        patches = self.patch(x)

        patches = patches + self.pos_embed

        x_masked, mask, ids_restore = self.random_mask(patches)

        latent = self.encoder(x_masked)

        latent = self.enc_to_dec(latent)

        N,L,D = latent.shape

        mask_tokens = self.mask_token.repeat(
            N,
            ids_restore.shape[1]-L,
            1
        )

        x_ = torch.cat([latent,mask_tokens],dim=1)

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1,1,D)
        )

        pred = self.decoder(x_)

        return pred,mask


# -----------------------------
# Unpatchify
# -----------------------------
def unpatchify(x):

    p = 16

    h = w = 14

    x = x.reshape(x.shape[0],h,w,p,p,3)

    x = x.permute(0,5,1,3,2,4)

    imgs = x.reshape(x.shape[0],3,h*p,w*p)

    return imgs


# -----------------------------
# Load Model
# -----------------------------
model = MAE()

model.load_state_dict(
    torch.load("mae_model.pth", map_location=device),
    strict=False
)
model.eval()
# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -----------------------------
# Reconstruction Function
# -----------------------------
def reconstruct(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        pred,mask = model(img)

        recon = unpatchify(pred)

    recon_img = recon[0].permute(1,2,0).numpy()

    recon_img = np.clip(recon_img,0,1)

    return recon_img


# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(

    fn=reconstruct,

    inputs=gr.Image(type="pil"),

    outputs=gr.Image(),

    title="Masked Autoencoder Image Reconstruction",

    description="Upload an image to see reconstruction using a Vision Transformer Masked Autoencoder."

)

interface.launch()
