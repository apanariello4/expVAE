from .OneClassMNIST import OneMNIST


class OneFMNIST(OneMNIST):
    urls = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"]


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    ds = OneFMNIST(
        root='/tmp/fmnist',
        one_class=3,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    a = next(iter(dl))
    torchvision.utils.save_image(a[0], './a.png')
