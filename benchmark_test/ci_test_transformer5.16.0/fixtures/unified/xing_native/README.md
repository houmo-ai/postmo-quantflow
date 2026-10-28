# Xing4.0 native CI source

These four unmodified configuration, model and tokenizer files come from
[XingChen-AGI/Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/tree/a32a2e99b5877c6b7ad16e9954ec08488bcc5fa2),
revision `a32a2e99b5877c6b7ad16e9954ec08488bcc5fa2`.
They match the original model source used by the Xing integration tests.

The upstream project distributes them under Apache-2.0. The license is retained
at `licenses/Xing4.0/LICENSE`, copied from the upstream GitHub repository at
revision `88b5308955ec9b072282ce0a7fd5d00e0029eb9c`.
See the Xing4.0 entry in `THIRD_PARTY_NOTICES`.

The Xing fixture replaces the model dimensions before construction and generates
random weights and a tiny SentencePiece vocabulary locally. No upstream weights
or tokenizer model are included. `XING4_0_MODEL_DIR` may override this source for
local compatibility checks; CI uses the bundled source without network access.
