import torch 

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
SEED = 1234

# define fields
IDX = 'idx'
PATTERN = 'pattern'
PERSON = 'person'
CUE_ID = 'cue_id'
SAR_ID = 'sar_id'
OBL_ID = 'obl_id'
ELI_ID = 'eli_id'
PERSPECTIVE = 'perspective'
CUE_TEXT = 'cue_text'
SAR_TEXT = 'sar_text'
OBL_TEXT = 'obl_text'
ELI_TEXT = 'eli_text'
CUE_USER = 'cue_user'
SAR_USER = 'sar_user'
OBL_USER = 'obl_user'
ELI_USER = 'eli_user'
LABEL = 'label'

# define tokens
PAD_TOKEN = '<PAD>'
END_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<SOS>'
SEP_TOKEN = '<SEP>'