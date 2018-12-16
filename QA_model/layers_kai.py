import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AttentionScore


class DocumentAttnOverAttn(nn.Module):
    def __init__(self, input_size, attention_hidden_size, num_head=5):
        super(DocumentAttnOverAttn, self).__init__()
        self.scoring = AttentionScore(input_size, attention_hidden_size)
        self.linear = nn.Linear(input_size * 2, input_size)
        self.fuse_layer = FusionLayer(input_size)
        self.multi_head = MultiHeadSelfAtt(input_size, num_head)
        self.input_gate = Gate(input_size)
        self.forget_gate = Gate(input_size)

    def forward(self, c1, c2, c_mask, flag):
        if flag == 0:
            return self.forward_cat(c1, c2, c_mask)
        elif flag == 1:
            return self.forward_fuse(c1, c2, c_mask)
        elif flag == 2:
            return self.forward_self_att(c1, c2, c_mask)
        elif flag == 3:
            return self.forward_gate(c1, c2, c_mask)
        elif flag == 4:
            return self.forward_forget(c1, c2, c_mask)
        elif flag == 5:
            return self.forward_add_gates(c1, c2, c_mask)
        elif flag == 6:
            return self.forward_simple_gates(c1, c2, c_mask)

    def forward_cat(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        a_mask = c_mask.unsqueeze(1).expand_as(scores)
        alpha_scores = scores.data.masked_fill(a_mask.data, -float('inf'))
        alpha = F.softmax(alpha_scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        b_mask = c_mask.unsqueeze(2).expand_as(scores)
        beta_scores = scores.data.masked_fill(b_mask.data, -float('inf'))
        beta = F.softmax(beta_scores, dim=1)
        aug_c1 = beta.transpose(1, 2).bmm(c2)

        return self.linear(torch.cat((aug_c2, aug_c1), dim=2))

    def forward_fuse(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        a_mask = c_mask.unsqueeze(1).expand_as(scores)
        alpha_scores = scores.data.masked_fill(a_mask.data, -float('inf'))
        alpha = F.softmax(alpha_scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        fused_c2 = self.fuse_layer(c2, aug_c2)
        return fused_c2

    def forward_self_att(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        mask = c_mask.unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        y = self.multi_head(aug_c2, c_mask)

        return y

    def forward_gate(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        a_mask = c_mask.unsqueeze(1).expand_as(scores)
        alpha_scores = scores.data.masked_fill(a_mask.data, -float('inf'))
        alpha = F.softmax(alpha_scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        fused_c2 = self.input_gate(aug_c2, c2)
        return fused_c2

    def forward_forget(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        a_mask = c_mask.unsqueeze(1).expand_as(scores)
        alpha_scores = scores.data.masked_fill(a_mask.data, -float('inf'))
        alpha = F.softmax(alpha_scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        fused_c2 = self.input_gate(aug_c2, c2)
        fused_c1 = self.forget_gate(c1, aug_c2)
        return self.linear(torch.cat([fused_c2, fused_c1], dim=2))

    def forward_add_gates(self, c1, c2, c_mask):
        scores = self.scoring(c2, c1)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        a_mask = c_mask.unsqueeze(1).expand_as(scores)
        alpha_scores = scores.data.masked_fill(a_mask.data, -float('inf'))
        alpha = F.softmax(alpha_scores, dim=2)
        aug_c2 = alpha.bmm(c1)

        fused_c2 = self.input_gate(aug_c2, c2)
        fused_c1 = self.forget_gate(c1, aug_c2)
        return fused_c1 + fused_c2

    def forward_simple_gates(self, c1, c2, c_mask):

        g2 = torch.sigmoid(c2)
        f2 = torch.tanh(c2)
        fused_c2 = g2 * f2

        g1 = torch.sigmoid(c1)
        f1 = torch.tanh(c1)
        fused_c1 = g1 * f1

        return fused_c1 + fused_c2


class MultiHeadSelfAtt(nn.Module):
    def __init__(self, input_size, num_head=5):
        super(MultiHeadSelfAtt, self).__init__()
        self.num_head = num_head
        self.multi_head = nn.ModuleList()
        self.size_list = [int(input_size / num_head)] * (num_head - 1)
        self.size_list.append(int(input_size - self.size_list[0] * (num_head - 1)))
        for i in range(num_head):
            self.multi_head.append(SelfAtt(self.size_list[i]))
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x, mask):
        x_s = torch.split(x, self.size_list, 2)
        output = []
        for i in range(self.num_head):
            output.append(self.multi_head[i](x_s[i], mask))
        y_c = torch.cat(output, dim=2)
        y = self.linear(y_c)
        return y


class SelfAtt(nn.Module):
    def __init__(self, input_size):
        super(SelfAtt, self).__init__()
        self.scoring = AttentionScore(input_size, input_size)

    def forward(self, x, mask):
        scores = self.scoring(x, x)

        diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
        scores.data.masked_fill_(diag_mask, -float('inf'))

        mask = mask.unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(mask.data, -float('inf'))

        alpha = F.softmax(scores, dim=2)
        y = alpha.bmm(x)

        return y


class Gate(nn.Module):
    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.linear_g = nn.Linear(input_size * 2, input_size)
        self.linear_f = nn.Linear(input_size * 2, input_size)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=2)
        gate = F.sigmoid(self.linear_g(z))
        f = F.tanh(self.linear_f(z))
        return gate * f


class FusionLayer(nn.Module):
    """
    vector based fusion
    m(x, y) = W([x, y, x * y, x - y]) + b
    g(x, y) = w([x, y, x * y, x - y]) + b
    :returns g(x, y) * m(x, y) + (1 - g(x, y)) * x
    """

    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.linear_f = nn.Linear(input_dim * 4, input_dim, bias=True)
        self.linear_g = nn.Linear(input_dim * 4, 1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        gated = self.sigmoid(self.linear_g(z))
        fusion = self.tanh(self.linear_f(z))
        return gated * fusion + (1 - gated) * x
