from .SRN_model import SRN_Net
from .SRNATT_model import SRNATT_Net
from .SRNATTS_model import SRNATTS_Net
from .UNet_model import U_Net
from .ATT_model import ATT_Net
from .ATT_Deblur_model import ATT_Deblur_Net


def get_model(args):
    if args.model == 'U_Net':
        return U_Net(args)
    elif args.model == 'SRN_Net':
        return SRN_Net(args)
    elif args.model == 'SRNATT_Net':
        return SRNATT_Net(args)
    elif args.model == 'SRNATTS_Net':
        return SRNATTS_Net(args)
    elif args.model == 'ATT_Net':
        return ATT_Net(args)
    elif args.model == 'ATT_Deblur_Net':
        return ATT_Deblur_Net(args)
    else:
        raise NotImplementedError("{} is not supported".format(args.model))