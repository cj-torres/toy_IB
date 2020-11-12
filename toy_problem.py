import torch


class Toy(torch.nn.Module):

    def __init__(self, tag_size):
        super().__init__()
        self.tags = torch.nn.Parameter(torch.rand(tag_size, 4), requires_grad=True)

    def forward(self, px, lam):
        pzgivenx = torch.nn.functional.softmax(self.tags, dim=0)
        #print("P(x): " + str(px))
        #print("P(z|x): " + str(pzgivenx))
        pz = pzgivenx.mv(px)
        #print("P(z): " + str(pz))

        pzx = prob_matrix(pzgivenx, px)
        #print("P(z,x): " + str(pzx))
        pzpx = torch.outer(pz, px)
        #print("P(z)P(x): " + str(pzpx))

        #pygivenz = torch.nn.functional.softmax(self.out, dim=0)
        py = torch.FloatTensor([px[:2].sum(),px[2:].sum()])
        #print("P(y): " + str(py))
        pyz = torch.stack([pzx[:,:2].sum(dim=1),pzx[:,2:].sum(dim=1)]).t()
        #print("P(y,z): " + str(pyz))
        pypz = torch.outer(pz,py)
        #print("P(y)P(z): " + str(pypz))

        #pypz = torch.cross(py, pz)

        mi_zx = mutual_information(pzpx, pzx)
        mi_yz = mutual_information(pypz, pyz)

        return ((-mi_yz + (lam*mi_zx)), mi_yz, mi_zx)

def mutual_information(pxpy, pxy):
    assert pxpy.size() == pxy.size()

    return (pxy*(torch.log2(pxy/pxpy))).sum()


def entropy(px):
    return -(torch.log(px) * px).sum()


def prob_matrix(A, v):
    return torch.einsum("ji,i->ji",A,v)


def train(tag_size, lam, epochs):
    model = Toy(tag_size)
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(0,epochs):
        (loss, myz, mzx) = model(torch.nn.functional.softmax(torch.rand(4),dim=0),lam)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        model.zero_grad()

        print(myz)
        print(mzx)

    print(torch.nn.functional.softmax(model.tags, dim=0))

train(8,1,10000)