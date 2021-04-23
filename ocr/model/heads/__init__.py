
__all__ = ['build_head']


def build_head(config, **kwargs):
    # det head
    from .det_db_head import DBHead
    from .det_east_head import EASTHead
    from .det_sast_head import SASTHead

    # rec head
    from .rec_ctc_head import CTCHead
    # from .rec_att_head import AttentionHead
    # from .rec_srn_head import SRNHead

    # cls head
    from .cls_head import ClsHead
    support_dict = [
        'DBHead', 'EASTHead', 'SASTHead', 'CTCHead', 'ClsHead'#, 'AttentionHead',
        #'SRNHead'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    print(config)
    module_class = eval(module_name)(**config, **kwargs)
    return module_class