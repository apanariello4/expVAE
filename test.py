import torch
from torchvision.utils import save_image
from model.vae_loss import vae_loss
from tqdm import tqdm


def save_one_recon_batch(model, device, test_loader, epoch):
    with torch.no_grad():
        for it, (x, _) in enumerate(test_loader):
            model.eval()
            x = x.to(device)

            x_hat, _, _ = model(x)

            imgs = torch.cat([x[0].transpose(0, 1), x_hat[0].transpose(0, 1)], dim=0)

            save_image(imgs, f'ep-{epoch}_recon_moving.png', nrow=10)

            break


def eval(model, device, test_loader):
    model.eval()
    test_loss = 0
    num_samples = 0
    with tqdm(total=len(test_loader), desc='Eval ') as pbar:
        with torch.no_grad():
            for data, _ in test_loader:

                data = data.to(device)

                batch_size = data.shape[0]
                num_samples += batch_size

                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                pbar.set_postfix(loss=test_loss / num_samples)
                pbar.update()

    test_loss /= len(test_loader.dataset)
    return test_loss


if __name__ == '__main__':
    from model.conv3dVAE import Conv3dVAE
    from utils.utils import load_moving_mnist, args
    arg = args()
    model = Conv3dVAE(latent_dim=512)
    checkpoint = torch.load('./checkpoints/moving_conv3dVAE_512_model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])

    _, test_loader = load_moving_mnist(arg)

    save_one_recon_batch(model, torch.device('cpu'), test_loader, checkpoint['epoch'])
