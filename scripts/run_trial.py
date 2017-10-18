from pytry import parser

from imem.model.trial import IMemTrial


trial = IMemTrial()

if __name__ == '__main__':
    args = parser.parse_args(trial, args=None, allow_filename=False)
    trial.run(**args)
