# https://help.aliyun.com/zh/marketplace/partner-driven-large-model-tokens-consumption-buried-point-proposal?spm=a2c4g.11186623.help-menu-30488.d_2_2_0.53a66065QheeNw&scm=20140722.H_2927986._.OR_help-T_cn~zh-V_1
# bury point header for aliyun
BURY_POINT_HEADER = {
    'x-dashscope-euid': '{"bizType":"B2B", "moduleType":"Third-partyproducts", "moduleCode":"market_91320506MACC8K9EXF", "accountType":"Aliyun", "accountId":""}'}


def get_base_url(credentials: dict, default_international: str = "https://dashscope-intl.aliyuncs.com/api/v1", default_domestic: str = "https://dashscope.aliyuncs.com/api/v1") -> str:
    """
    获取 base URL，优先级：自定义 base_url > use_international_endpoint > 默认国内端点
    
    :param credentials: 凭证字典
    :param default_international: 默认国际端点
    :param default_domestic: 默认国内端点
    :return: base URL
    """
    # 如果提供了自定义 base_url，优先使用
    if credentials.get("base_url"):
        return credentials["base_url"]
    
    # 否则根据 use_international_endpoint 选择
    if credentials.get("use_international_endpoint", "false") == "true":
        return default_international
    
    return default_domestic


def get_compatible_base_url(credentials: dict) -> str:
    """
    获取兼容模式的 base URL（用于 OpenAI 客户端）
    
    :param credentials: 凭证字典
    :return: 兼容模式的 base URL
    """
    default_international = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    default_domestic = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return get_base_url(credentials, default_international, default_domestic)