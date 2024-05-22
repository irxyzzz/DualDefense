import warnings

import tenseal as ts
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


def get_ckks_context() -> ts.Context:
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context


def ckks_consine_similarity(
    context: ts.Context, t1: torch.Tensor, t2: torch.Tensor
) -> float:
    t1_norm = t1 / torch.norm(t1)
    t2_norm = t2 / torch.norm(t2)

    list1 = t1_norm.flatten().tolist()
    list2 = t2_norm.flatten().tolist()

    t1_encrypted = ts.ckks_vector(context, list1)
    t2_encrypted = ts.ckks_vector(context, list2)
    cosine_similarity_encrypted = t1_encrypted.dot(t2_encrypted)

    return cosine_similarity_encrypted


context_ckks = get_ckks_context()
