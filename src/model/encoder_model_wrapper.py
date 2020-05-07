from model.encoder_model import *


class EncoderModelWrapper:
    """ Wrapper for encoder model """

    @staticmethod
    def get_encoder_model(model_type, config, constants, bootstrap_model=None):

        if model_type == "backwardmodel":
            return BackwardEncoderModel(config, constants, bootstrap_model)
        elif model_type == "forwardmodel":
            return ForwardEncoderModel(config, constants, bootstrap_model)
        else:
            raise NotImplementedError()
