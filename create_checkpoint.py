import torch
from train import DeepInfantModel

def main():
    model = DeepInfantModel()
    torch.save(model.state_dict(), 'deepinfant.pth')
    print('Saved untrained checkpoint to deepinfant.pth')

if __name__=='__main__':
    main()
