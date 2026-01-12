from typing import Optional

from .remote import LLMRemoteModel


def load_llm_model(
        config_file_or_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None,
        **params,
):
    from ..manage import LLMConfig

    try:
        llm_config = LLMConfig.open(config_file_or_dir or '.')
    except FileNotFoundError:
        llm_config = None

    if llm_config:
        try:
            llm_params = llm_config.get_model_params(model_name=model_name, **params)
        except KeyError:
            llm_params = None
    else:
        llm_params = None

    if llm_params is not None:
        # known model is found or generated from the config file
        if base_url:
            llm_params['base_url'] = base_url
        if api_token:
            llm_params['api_token'] = api_token
        llm_params.update(**params)

    elif base_url:
        # newly generated llm config
        llm_params = {'base_url': base_url}
        if api_token is None:
            raise ValueError(f'API token must be specified, but {api_token!r} found.')
        llm_params['api_token'] = api_token
        if not model_name:
            raise ValueError(f'Model name must be non-empty, but {model_name!r} found.')
        llm_params['model_name'] = model_name
        llm_params.update(**params)

    else:
        raise RuntimeError('No model parameters specified and no local configuration for falling back.')

    return LLMRemoteModel(**llm_params)
