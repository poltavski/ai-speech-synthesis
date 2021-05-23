import json
import os
import re
import torch
import warnings


def init_models(
    device=torch.device("cuda"),
    speaker_list: list = ["dina", "mila", "tisha", "pasha", "tina", "nika"],
):
    models = {
        speaker: fetch_model(speaker=speaker).to(device) for speaker in speaker_list
    }
    return models


def init_config(config_file="models.json"):
    with open(config_file) as config_data:
        config = json.load(config_data)
        return config["vocab"], config["rate"], config["output_dir"]


def fetch_model(speaker="mila", config_file="models.json"):
    with open(config_file) as config_data:
        config = json.load(config_data)
        model_names = config["names"]
        model_path = f"{config['dir']}/{speaker}.{config['ext']}"
        model = init_jit_model(model_path) if speaker in model_names else None
        return model


def init_jit_model(model_path: str, device: torch.device = torch.device("cuda")):
    torch.set_grad_enabled(False)
    # model_dir = os.path.dirname(__file__)
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, os.path.basename(model_path))
    if os.path.isfile(model_path):
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    else:
        model = None
    return model


def prepare_text_input(text, symbols, symbol_to_id=None):
    if len(text) > 140:
        warnings.warn("Text string is longer than 140 symbols.")

    if symbol_to_id is None:
        symbol_to_id = {s: i for i, s in enumerate(symbols)}

    text = text.lower()
    text = re.sub(r"[^{}]".format(symbols[2:]), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text[-1] not in [".", "!", "?"]:
        text = text + "."
    text = text + symbols[1]

    text_ohe = [symbol_to_id[s] for s in text if s in symbols]
    text_tensor = torch.LongTensor(text_ohe)
    return text_tensor


def prepare_tts_model_input(text: str or list, symbols: str):
    if type(text) == str:
        text = [text]
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    if len(text) == 1:
        return (
            prepare_text_input(text[0], symbols, symbol_to_id).unsqueeze(0),
            torch.LongTensor([0]),
        )

    text_tensors = []
    for string in text:
        string_tensor = prepare_text_input(string, symbols, symbol_to_id)
        text_tensors.append(string_tensor)
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(t) for t in text_tensors]), dim=0, descending=True
    )
    max_input_len = input_lengths[0]
    batch_size = len(text_tensors)

    text_padded = torch.ones(batch_size, max_input_len, dtype=torch.int32)

    for i, idx in enumerate(ids_sorted_decreasing):
        text_tensor = text_tensors[idx]
        in_len = text_tensor.size(0)
        text_padded[i, :in_len] = text_tensor

    return text_padded, ids_sorted_decreasing


def process_tts_model_output(out, out_lens, ids, sample_rate):
    assert sample_rate in [8000, 16000]
    out = out.to("cpu")
    out_lens = out_lens.to("cpu")
    _, orig_ids = ids.sort()

    proc_outs = []
    srf = 2 if sample_rate == 16000 else 1
    orig_out = out.index_select(0, orig_ids)
    orig_out_lens = out_lens.index_select(0, orig_ids)

    for i, out_len in enumerate(orig_out_lens):
        proc_outs.append(orig_out[i][: out_len * srf])
    return proc_outs


def apply_tts(
    texts: list,
    model: torch.nn.Module,
    sample_rate: int,
    symbols: str,
    device: torch.device,
):
    text_padded, orig_ids = prepare_tts_model_input(texts, symbols=symbols)
    out, out_lens = model(text_padded.to(device))
    audios = process_tts_model_output(out, out_lens, orig_ids, sample_rate)
    return audios
