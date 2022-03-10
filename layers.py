"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, char_drop_prob, use_char_emb=False):
        super(EmbeddingWithChar, self).__init__()
        self.use_char_emb = use_char_emb
        self.drop_prob = drop_prob
        #New
        self.char_drop_prob = char_drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        char_embed_size = char_vectors.size(1)
        
        if use_char_emb:
            self.conv2d = nn.Conv2d(char_embed_size, hidden_size, kernel_size = (1,5))
            self.hwy = HighwayEncoder(2, hidden_size)
            self.proj = nn.Linear(word_vectors.size(1)+hidden_size, hidden_size, bias=False)
        else:
            self.hwy = HighwayEncoder(2, hidden_size)
            self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)

        

    def forward(self, x, y=None):
        word_emb = self.embed(x)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        
        if self.use_char_emb:
            # (batch_size, seq_len, max_char, char_dim)
            char_emb = self.char_embed(y)
            # (batch_size, char_dim, max_char, seq_len)
            char_emb = torch.transpose(char_emb, 1, 3)
            # (batch_size, char_dim, seq_len, max_char)
            char_emb = torch.transpose(char_emb, 2, 3)
            # char_drop_prob = 0.05
            char_emb = F.dropout(char_emb, self.char_drop_prob, self.training)
            char_emb = self.conv2d(char_emb) 
            
            char_emb = F.relu(char_emb)
            char_emb = torch.transpose(char_emb, 1, 2)
            # apply maxpool 1d on the last dimension, kernel_size = max num of char per word
            char_emb, _ = torch.max(char_emb, dim=3)
            # want char_emb to be (batch_size, seq_len, hidden_size)
            emb = torch.cat([word_emb, char_emb], dim=-1)
        else:
            emb = word_emb

    
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb



class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class CoAttention(nn.Module):
    """Coattention as described in "Dynamic Coattention Networks".
    TODO: try expanding the size of the coattended output to see if we get better results (e.g. expand the output dim)
    A two-way attention between context and question. Performs a second level
    attention computation - attending over representations that are attention outputs.


    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        pass
        super(CoAttention, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        #self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        # apply a linear layer to q 
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.c_sentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_sentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        
        for weight in (self.c_sentinel, self.q_sentinel):
            nn.init.xavier_uniform_(weight)
        
        #TODO:check dimensions
        self.biLSTM = torch.nn.LSTM(2*self.hidden_size, self.hidden_size, bidirectional=True)

        self.dropout = torch.nn.Dropout(p=self.drop_prob)

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()

        q_len = q.size(1)

        # perform one linear layer on q
        q_prime = self.q_proj(q)
        q_prime = torch.tanh(q_prime)

        # TODO: concat all cs together
        cs = torch.cat((c, self.c_sentinel.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2)), dim=1)
        # TODO: concat all qs together
        qs = torch.cat((q_prime, self.q_sentinel.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2)), dim=1)

        # get an affinity matrix
        L = self.get_affinity_matrix(cs, qs) # (c_len + 1, q_len + 1)

        #append additional vector of masks to c_mask for c_sentinel
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        c_mask = torch.cat((c_mask, c_mask[:,-2:-1,:]), dim=1)

        #append additional vector of masks to q_mask for q_sentinel
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        q_mask = torch.cat((q_mask, q_mask[:, :, -2:-1]), dim=2)

        alphas = masked_softmax(L, q_mask, dim=2)       # (batch_size, c_len, q_len) 
        betas = masked_softmax(L, c_mask, dim=1)       # (batch_size, c_len, q_len)  
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        
        a = torch.bmm(alphas, qs)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(betas.transpose(1, 2), cs)

        s = torch.bmm(alphas, b)

        concatted_s_a = torch.cat((s, a), dim=2)
        # currently LSTM is input_dim hidden_size, output_dim hidden_size
        x, _ = self.biLSTM(concatted_s_a)    

        x = self.dropout(x)
        x = x[:, :-1, :]

        return x
    
    def get_affinity_matrix(self, cs, qs):

        L = torch.bmm(cs, qs.transpose(1,2))

        return L 

class SelfAttention(nn.Module):
    """Self attention, or self-matching attention, as described in the paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_dim, hidden_size, drop_prob=0.1, layers=2):
        pass
        super(SelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.layers = layers # how many layers should we apply?
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        
        # build linear layers -- follow self attention strategy given in the paper, check that this makes sense
        
        self.W1 = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        self.W2 = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.g = nn.Sequential(nn.Linear(2*self.input_dim, 2*self.input_dim, bias=False), nn.Sigmoid()) # do we need sigmoid here??
        self.rnn = nn.GRU(input_dim*2, self.hidden_size, bidirectional=True, num_layers=3, dropout=self.drop_prob) # can we expand the size of th emodel in any way? 

    def forward(self, prev_att):
        
        p = prev_att
        # a little bit of dropout before this attention layer; finding higher performance without extra dropout
        # p = F.dropout(p, self.drop_prob, self.training)
        

        #use W1
        W1 = self.W1(p).repeat(p.size(0), 1, 1, 1)
        W2 = self.W2(p).repeat(p.size(0), 1, 1, 1)
        p_long = p.repeat(p.size(0), 1, 1, 1)
        s = self.tanh(W1 + W2)
        s = self.V(s)
        a = self.softmax(s)

        #get c by first combining a and p, then applying self-gating
        c = (a * p_long)
        c = torch.sum(c, dim=0)
        c = torch.cat((p, c), dim=2)
        c = torch.mul(c, self.g(c))
        # c: get sum
        output, _ = self.rnn(c)
        return F.dropout(output, self.drop_prob, self.training)



class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class BiDAFCoattendedOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFCoattendedOutput, self).__init__()
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

