import torch
import torch.nn as nn


class NPUnit(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size):
        super(NPUnit, self).__init__()
        self.padding = int((kernel_size[0]-1)/2)
        stride=1
        num_hidden=out_channels
        self.num_hidden=out_channels
        filter_size=kernel_size
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x_t, h_t, c_t,m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
       # print(f_t.shape,c_t.shape,i_t.shape,g_t.shape)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn

# class SelfAttention(nn.Module):
#     def __init__(self, num_hidden):
#         super(SelfAttention, self).__init__()
#         self.num_hidden = num_hidden
#         self.q_linear = nn.Linear(num_hidden, num_hidden)
#         self.k_linear = nn.Linear(num_hidden, num_hidden)
#         self.v_linear = nn.Linear(num_hidden, num_hidden)
#         self.out_linear = nn.Linear(num_hidden, num_hidden)

#     def forward(self, x):
#         q = self.q_linear(x)  # Apply linear transformation
#         k = self.k_linear(x)  # Apply linear transformation
#         v = self.v_linear(x)  # Apply linear transformation

#         # Perform scaled dot-product attention
#         attention_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.num_hidden ** 0.5)
#         attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
#         out = torch.matmul(attention_weights, v)

#         # Apply another linear transformation for the output
#         out = self.out_linear(out)

#         return out

# class NPUnit(nn.Module):
#     def __init__(self, in_channel, out_channels):
#         super(NPUnit, self).__init__()
#         self.num_hidden = out_channels
#         self.conv_x = nn.Sequential(
#             nn.Conv2d(in_channel, self.num_hidden * 7, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#         self.conv_last = nn.Conv2d(self.num_hidden * 2, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
#         self.self_attention = SelfAttention(self.num_hidden)

#     def forward(self, x_t, h_t, c_t, m_t):
#         x_concat = self.conv_x(x_t)
#         i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)

#         i_t = torch.sigmoid(i_x)
#         f_t = torch.sigmoid(f_x)
#         g_t = torch.tanh(g_x)

#         c_new = f_t * c_t + i_t * g_t

#         i_t_prime = torch.sigmoid(i_x_prime)
#         f_t_prime = torch.sigmoid(f_x_prime)
#         g_t_prime = torch.tanh(g_x_prime)

#         m_new = f_t_prime * m_t + i_t_prime * g_t_prime

#         mem = torch.cat((c_new, m_new), 1)
#         self_attention_out = self.self_attention(mem)
#         h_new = torch.tanh(self.conv_last(self_attention_out))

#         return h_new, c_new, m_new
