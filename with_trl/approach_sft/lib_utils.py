
def get_is_encoder_decoder(
    model_name_or_path: str
) -> bool:
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)  # type: ignore
    return config.is_encoder_decoder
