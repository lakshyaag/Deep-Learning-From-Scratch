import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune=False, debug=False):
        super(Encoder, self).__init__()

        resnet = torchvision.models.resnet101(weights="IMAGENET1K_V1")

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        for param in self.resnet.parameters():
            param.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        self._debug_print(x, "[ENCODER] Input")
        # B x 3 x H x W -> B x 2048 x H/32 x W/32
        x = self.resnet(x)
        self._debug_print(x, "[ENCODER] Resnet")
        # B x 2048 x H/32 x W/32 -> B x 2048 x ENCODED_IMAGE_SIZE x ENCODED_IMAGE_SIZE
        x = self.adaptive_pool(x)
        self._debug_print(x, "[ENCODER] Adaptive Pool")
        # B x 2048 x ENCODED_IMAGE_SIZE x ENCODED_IMAGE_SIZE -> B x ENCODED_IMAGE_SIZE x ENCODED_IMAGE_SIZE x 2048
        x = x.permute(0, 2, 3, 1)
        self._debug_print(x, "[ENCODER] Permute")
        return x


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, debug=False):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.debug = debug

        self.init_weights()

    def init_weights(self):
        self.encoder_att.weight.data.uniform_(-0.1, 0.1)
        self.encoder_att.bias.data.fill_(0)

        self.decoder_att.weight.data.uniform_(-0.1, 0.1)
        self.decoder_att.bias.data.fill_(0)

        self.full_att.weight.data.uniform_(-0.1, 0.1)
        self.full_att.bias.data.fill_(0)

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, image_features, hidden_state):
        """
        image_features: B x PIXEL_SIZE x ENCODER_DIM
        hidden_state: B x DECODER_DIM
        """

        # B x PIXEL_SIZE x ENCODER_DIM -> B x PIXEL_SIZE x ATTENTION_DIM
        encoder_att = self.encoder_att(image_features)
        self._debug_print(encoder_att, "[ATTENTION] Encoder Attention")

        # B x DECODER_DIM -> B x ATTENTION_DIM
        decoder_att = self.decoder_att(hidden_state)
        self._debug_print(decoder_att, "[ATTENTION] Decoder Attention")

        # (B x PIXEL_SIZE x ATTENTION_DIM) + (B x 1 x ATTENTION_DIM) -> B x PIXEL_SIZE x ATTENTION_DIM -> B x PIXEL_SIZE
        attention_score = self.full_att(
            self.relu(encoder_att + decoder_att.unsqueeze(1))
        ).squeeze(2)
        self._debug_print(attention_score, "[ATTENTION] Attention Score")

        # B x PIXEL_SIZE -> B x PIXEL_SIZE
        alpha = self.softmax(attention_score)
        self._debug_print(alpha, "[ATTENTION] Alpha")

        # B x PIXEL_SIZE x ENCODER_DIM (.) B x PIXEL_SIZE x 1 -> B x ENCODER_DIM
        awe = (image_features * alpha.unsqueeze(2)).sum(dim=1)
        self._debug_print(awe, "[ATTENTION] Attention Weighted Image Features")

        return awe, alpha


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        decoder_dim,
        attention_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
        debug=False,
    ):
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim, debug=debug)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        self.decoder_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def init_hidden_state(self, image_features):
        """
        image_features: B x PIXEL_SIZE x 2048
        """

        mean_image_features = image_features.mean(dim=1)

        # B x 2048 -> B x DECODER_DIM
        h = self.init_h(mean_image_features)
        c = self.init_c(mean_image_features)

        return h, c

    def forward(self, image_features, caption_tokens, caption_lengths):
        """

        image_features: B x ENCODED_IMAGE_SIZE x ENCODED_IMAGE_SIZE x 2048
        caption_tokens: B x CAPTION_LENGTH
        caption_lengths: B
        """

        batch_size = image_features.size(0)
        encoder_dim = image_features.size(-1)

        vocab_size = self.vocab_size
        device = image_features.device

        self._debug_print(image_features, "[DECODER] Image Features")

        image_features = image_features.view(batch_size, -1, encoder_dim)
        pixel_size = image_features.size(1)
        self._debug_print(image_features, "[DECODER] Flattened Image Features")

        self._debug_print(caption_tokens, "[DECODER] Caption Tokens")
        self._debug_print(caption_lengths, "[DECODER] Caption Length")
        caption_lengths, sort_index = caption_lengths.sort(dim=0, descending=True)
        image_features = image_features[sort_index]
        caption_tokens = caption_tokens[sort_index]

        # B x MAX_CAPTION_LENGTH -> B x MAX_CAPTION_LENGTH x EMBED_DIM
        embeddings = self.embedding(caption_tokens)
        self._debug_print(embeddings, "[DECODER] Embeddings")

        h, c = self.init_hidden_state(image_features)

        decode_lengths = (caption_lengths - 1).tolist()

        # B x MAX_CAPTION_LENGTH - 1 x VOCAB_SIZE
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), pixel_size).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([length > t for length in decode_lengths])

            awe, alpha = self.attention(image_features[:batch_size_t], h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            awe = gate * awe

            if t == 1:
                self._debug_print(awe, "[DECODER] Attention Weighted Image Features")
                self._debug_print(alpha, "[DECODER] Alpha")

                self._debug_print(h, "[DECODER] Hidden State")
                self._debug_print(c, "[DECODER] Cell State")

            h, c = self.decoder_step(
                torch.cat([embeddings[:batch_size_t, t, :], awe], dim=1),
                (h[:batch_size_t], c[:batch_size_t]),
            )

            preds = self.fc(self.dropout(h))

            if t == 1:
                self._debug_print(preds, "[DECODER] Prediction")

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, caption_tokens, decode_lengths, alphas, sort_index


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        encoded_image_size,
        embed_dim,
        decoder_dim,
        attention_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
        fine_tune=False,
        debug=False,
    ):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = Encoder(
            encoded_image_size=encoded_image_size, fine_tune=fine_tune, debug=debug
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout,
            debug=debug,
        )

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, image, caption_tokens, caption_lengths):
        self._debug_print(image, "[MODEL] Image")
        self._debug_print(caption_tokens, "[MODEL] Caption Tokens")
        self._debug_print(caption_lengths, "[MODEL] Caption Lengths")

        image_features = self.encoder(image)
        self._debug_print(image_features, "[MODEL] Image Features")

        predictions, caption_tokens, decode_lengths, alphas, sort_index = self.decoder(
            image_features, caption_tokens, caption_lengths
        )
        self._debug_print(predictions, "[MODEL] Predictions")

        return (
            image_features,
            predictions,
            caption_tokens,
            decode_lengths,
            alphas,
            sort_index,
        )

    def generate_caption_beam_search(self, image, beam_size=5, max_caption_length=20):
        k = beam_size

        device = self.decoder.fc.weight.device

        image_features = self.encoder(image)
        encoded_image_size = image_features.size(1)
        encoder_dim = image_features.size(-1)

        image_features = image_features.view(1, -1, encoder_dim)
        pixel_size = image_features.size(1)

        self._debug_print(image_features, "[BEAM SEARCH] Image Features")

        image_features = image_features.expand(k, pixel_size, encoder_dim)
        self._debug_print(image_features, "[BEAM SEARCH] Image Features - Expanded")

        k_prev_words = torch.LongTensor([[self.tokenizer.stoi["<SOS>"]]] * k).to(device)
        self._debug_print(k_prev_words, "[BEAM SEARCH] Prev Words")

        seqs = k_prev_words

        top_k_scores = torch.zeros(k, 1).to(device)

        seqs_alpha = torch.ones(k, 1, encoded_image_size, encoded_image_size).to(device)

        complete_seqs = list()
        complete_seqs_alphas = list()
        complete_seqs_scores = list()

        step = 1
        h, c = self.decoder.init_hidden_state(image_features)

        self._debug_print(h, "[BEAM SEARCH] Initial hidden State")
        self._debug_print(c, "[BEAM SEARCH] Initial Cell State")

        while True:
            # print("-" * 50)
            # print(f"Step: {step}")
            # K x 1 -> K x 1 x EMBED_DIM -> K x EMBED_DIM
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
            # self._debug_print(embeddings, "[BEAM SEARCH] Embeddings")

            # (K x PIXEL_SIZE x ENCODER_DIM, K x DECODER_DIM) -> K x ENCODER_DIM
            awe, alpha = self.decoder.attention(image_features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            awe = gate * awe
            # self._debug_print(awe, "[BEAM SEARCH] Attention Weighted Image Features")

            alpha = alpha.view(-1, encoded_image_size, encoded_image_size)
            # self._debug_print(alpha, "[BEAM SEARCH] Alpha")

            # K x EMBED_DIM -> (K x DECODER_DIM, K x DECODER_DIM)
            h, c = self.decoder.decoder_step(
                torch.cat([embeddings, awe], dim=1), (h, c)
            )

            # self._debug_print(h, "[BEAM SEARCH] Hidden State")
            # self._debug_print(c, "[BEAM SEARCH] Cell State")

            # K x DECODER_DIM -> K x VOCAB_SIZE
            scores = self.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            # self._debug_print(scores, "[BEAM SEARCH] Predicted scores")

            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)

            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            # print(f"Top K Words: {top_k_words}")
            # print(f"Top K Scores: {top_k_scores}")

            prev_word_inds = top_k_words // self.vocab_size
            next_word_inds = top_k_words % self.vocab_size

            # self._debug_print(prev_word_inds, "[BEAM SEARCH] Prev Word Indices")
            # self._debug_print(next_word_inds, "[BEAM SEARCH] Next Word Indices")

            # print(f"Prev Word Indices: {prev_word_inds}")
            # print(f"Next Word Indices: {next_word_inds}")

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            self._debug_print(seqs, "[BEAM SEARCH] Sequences")
            # print(f"Sequences: {seqs}")

            seqs_alpha = torch.cat(
                [seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1
            )

            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != self.tokenizer.stoi["<EOS>"]
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # print(f"Incomplete Indices: {incomplete_inds}")
            # print(f"Complete Indices: {complete_inds}")

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alphas.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)

            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]

            image_features = image_features[prev_word_inds[incomplete_inds]]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # print(f"Top K Scores: {top_k_scores}")
            # print(f"K Prev Words: {k_prev_words}")

            if step > max_caption_length:
                break

            step += 1

        # If no sequences are completed, return all sequences
        if len(complete_seqs_scores) == 0:
            return seqs, seqs_alpha

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alphas[i]

        return seq, alphas
