import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from pprint import pprint
from tqdm import tqdm
from losses_kd import DistillationLoss
import os
from config import get_config_parser
import json

from ViT_class import PromptedVisionTransformer

# Set the GPU index to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(epoch, model, train_loader, device, optimizer, num_epochs, loss_func, objective):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        if objective == 'teacher' or objective == 'student':
            outputs = model(images)
            loss = loss_func(outputs, labels)
        elif objective == 'distillation':
            outputs, student_weights = model(images)
            loss, loss_base, loss_distill = loss_func(images, outputs, labels, student_weights)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i + 1) % 50 == 0:
            if objective == 'distillation':
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_base: {:.4f}, Loss_distill: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), loss_base.item(),
                              loss_distill.item()))
            else:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    return loss.item()


def validate(epoch, model, val_loader, device, num_epochs, val_dataset, objective):
    # Evaluate the model on validation dataset
    with torch.no_grad():
        # Set model to evaluation mode
        model.eval()

        # Compute validation accuracy
        correct = 0
        for images, labels in tqdm(val_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            if objective == 'teacher' or objective == 'student':
                outputs = model(images)
            elif objective == 'distillation':
                outputs, _ = model(images)

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / len(val_dataset)
        print('Epoch [{}/{}], Validation Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, val_accuracy))
        return val_accuracy


def main():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = get_config_parser()
    args = parser.parse_args()

    if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load CIFAR dataset
        if args.dataset == "CIFAR-10":
            train_dataset = datasets.CIFAR10(root='~/datasets', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10(root='~/datasets', train=False, download=True, transform=transform_val)
            num_class = 10
        else:
            train_dataset = datasets.CIFAR100(root='~/datasets', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100(root='~/datasets', train=False, download=True, transform=transform_val)
            num_class = 100

    elif args.dataset == "Flowers-102":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # load Flowers dataset
        train_dataset = datasets.Flowers102(root='~/datasets', split="train", download=True, transform=transform_train)
        val_dataset = datasets.Flowers102(root='~/datasets', split="val", download=True, transform=transform_val)
        num_class = 102

    # Create data loaders for training and validation datasets
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # "teacher", "student", "distillation"
    if args.objective == 'teacher':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_class)
        for name, param in model.named_parameters():
            if not ('head' in name):
                param.requires_grad = False
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.objective == 'student':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_class)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.objective == 'distillation':
        model_t = PromptedVisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                                                num_classes=num_class)
        pretrained_path = './weights/' + f'{args.dataset}_{args.objective}.pth'
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"The file {pretrained_path} does not exist! Please train the teacher model on"
                                    f" the target dataset first.")
        pretrained_dict = torch.load(pretrained_path)
        model_t.load_state_dict(pretrained_dict)

        pretrained_s = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_class)
        model_s = PromptedVisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                                            num_classes=num_class)
        pretrained_state = pretrained_s.state_dict()
        model_s_state = model_s.state_dict()
        model_s_state.update(pretrained_state)
        model_s.load_state_dict(model_s_state)

        model_t.to(device)
        model_s.to(device)
        optimizer = optim.Adam(model_s.parameters(), lr=args.lr)

    CE = nn.CrossEntropyLoss()
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    for epoch in range(args.epochs):
        if args.objective == 'distillation':
            KDloss_func = DistillationLoss(base_criterion=CE, teacher_model=model_t, alpha=0.5, tau=2,
                                           distillation=args.distillation)
            train_loss = train(epoch, model_s, train_loader, device, optimizer, args.epochs, KDloss_func, args.objective)
            valid_acc = validate(epoch, model_s, val_loader, device, args.epochs, val_dataset, args.objective)
        else:
            train_loss = train(epoch, model, train_loader, device, optimizer, args.epochs, CE, args.objective)
            valid_acc = validate(epoch, model, val_loader, device, args.epochs, val_dataset, args.objective)

        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        if epoch == 0:
            best_val = valid_acc
        if best_val <= valid_acc:
            best_val = valid_acc
            print('SAVED weights in Epoch {:d}'.format(epoch))
            if not os.path.exists('./weights'):
                os.mkdir('./weights')
            if args.objective == 'distillation':
                save_path = f'{args.dataset}_{args.objective}'
                torch.save(model_s.state_dict(),
                           './weights/' + save_path + '.pth')
            elif args.objective == 'teacher':
                save_path = f'{args.dataset}_{args.objective}'
                torch.save(model.state_dict(),
                           './weights/' + save_path + '.pth')
            elif args.objective == 'student':
                save_path = f'{args.dataset}_{args.objective}'
                torch.save(model.state_dict(),
                           './weights/' + save_path + '.pth')

        # Save log
        if args.logdir is not None:
            print(f'Writing training logs to {args.logdir}...')
            os.makedirs(args.logdir, exist_ok=True)
            with open(os.path.join(args.logdir, save_path + '.json'), 'w') as f:
                f.write(json.dumps(
                    {
                        "train_losses": train_losses,
                        # "valid_losses": valid_losses,
                        # "train_accs": train_accs,
                        "valid_accs": valid_accs,
                        # "train_times": train_times,
                        # "valid_times": valid_times,
                        # "test_loss": test_loss,
                        # "test_acc": test_acc
                    },
                    indent=4,
                ))


if __name__ == '__main__':
    main()
