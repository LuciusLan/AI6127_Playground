### Codes for the AI6127 course at NTU

#### lang_model:

A simple language model using basic feed-forward neural network, as proposed in the *A neural probabilistic language model* paper [(Bengio, et. al. 2013)](#references)

#### NER:

CNN_BiLSTM_CRF NER tagging system as proposed by [Ma and Hovy. 2016](#references)

Implementation of which was based on https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial

#### CSNER:

Code-Switched NER, following the 2018 ACL CS workshop's setting

Implementation based on the NER model and edited to follow [Wang et. al](#references) to enhance with attention

Planned to test also sub-word level encoding (BPE)

CRF implementation was based on https://github.com/allanj/pytorch_lstmcrf

(Actually I could've clone the whole structure from it... But I decide to give myself more challenge to rewrite the original model with batch processing, in order to understand PyTorch better)

#### References

[1] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic
language model. J. Mach. Learn. Res., 3:1137–1155, March 2003.

[2] Xuezhe Ma and Eduard Hovy. End-to-end sequence labeling via bi-directional LSTM-CNNsCRF. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 1064–1074, Berlin, Germany, August 2016. Association for
Computational Linguistics.

[3] Wang, C., Cho, K., & Kiela, D. (2018, July). Code-switched named entity recognition with embedding attention. In Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching (pp. 154-158).