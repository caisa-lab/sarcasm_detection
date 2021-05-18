from enum import Enum

class GraphTypes(Enum):
    ONLY_ALL_TWEETTYPES = 'allTweetTypes'
    FULL = 'full'
    TWEETSONLY_NOCUE = 'tweetsOnly_noCue'
    ONLY_SAR = 'onlySar'
    FULL_WITH_CUE = 'full_with_cue'
    WITH_ELICIT = 'with_elicit'
    WITH_OBL = 'with_obl'
    
    
class ColNames(Enum):
    FULL =  { 'user': ['sar_user', 'obl_user', 'eli_user'],
             'tweet':  ['sar_text', 'obl_text', 'eli_text'],
             'tweet_id': ['sar_id', 'obl_id', 'eli_id'],
             'cols': ['sar', 'obl', 'eli'] }
    
    FULL_WITH_CUE = { 'user': ['cue_user', 'sar_user', 'obl_user', 'eli_user'],
             'tweet':  ['cue_text', 'sar_text', 'obl_text', 'eli_text'],
             'tweet_id': ['cue_id', 'sar_id', 'obl_id', 'eli_id'],
             'cols': ['cue', 'sar', 'obl', 'eli'] }
    
    ONLY_SAR = { 'user': ['sar_user'],
             'tweet':  ['sar_text'],
             'tweet_id': ['sar_id'],
             'cols': ['sar'] }
    
    WITH_ELICIT = { 'user': ['sar_user', 'eli_user'],
             'tweet':  ['sar_text', 'eli_text'],
             'tweet_id': ['sar_id', 'eli_id'],
             'cols': ['sar', 'eli'] }
    
    WITH_OBL = { 'user': ['sar_user', 'obl_user'],
             'tweet':  ['sar_text', 'obl_text'],
             'tweet_id': ['sar_id', 'obl_id'],
             'cols': ['sar', 'obl'] }

    
class GraphTypeToColNames:
    @staticmethod
    def get_col_names(graphType):
        if graphType is GraphTypes.FULL or graphType is GraphTypes.TWEETSONLY_NOCUE:
            return ColNames.FULL.value
        elif graphType is GraphTypes.FULL_WITH_CUE:
            return ColNames.FULL_WITH_CUE.value
        elif graphType is GraphTypes.ONLY_SAR:
            return ColNames.ONLY_SAR.value
        elif graphType is GraphTypes.WITH_ELICIT:
            return ColNames.WITH_ELICIT.value
        elif graphType is GraphTypes.WITH_OBL:
            return ColNames.WITH_OBL.value
        elif graphType is GraphTypes.ONLY_ALL_TWEETTYPES:
            return ColNames.FULL.value