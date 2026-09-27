import torch
import clip

def text_prompt(data):
    # text_aug = ['{}']
    text_aug = 'This is a video about {}'
    # data.classes: [num_classes, 77]
    classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return classes

def text_prompt_ensemble(data):
    # text_aug = ['{}']
    text_dict = {}
    # text_aug = [
    #             f'a photo of {{}}.',
    #             f'a photo of a person {{}}.',
    #             f'a photo of a person using {{}}.',
    #             f'a photo of a person doing {{}}.',
    #             f'a photo of a person during {{}}.',
    #             f'a photo of a person performing {{}}.',
    #             f'a photo of a person practicing {{}}.',
    #             f'a video of {{}}.',
    #             f'a video of a person {{}}.',
    #             f'a video of a person using {{}}.',
    #             f'a video of a person doing {{}}.',
    #             f'a video of a person during {{}}.',
    #             f'a video of a person performing {{}}.',
    #             f'a video of a person practicing {{}}.',
    #             f'a example of {{}}.',
    #             f'a example of a person {{}}.',
    #             f'a example of a person using {{}}.',
    #             f'a example of a person doing {{}}.',
    #             f'a example of a person during {{}}.',
    #             f'a example of a person performing {{}}.',
    #             f'a example of a person practicing {{}}.',
    #             f'a demonstration of {{}}.',
    #             f'a demonstration of a person {{}}.',
    #             f'a demonstration of a person using {{}}.',
    #             f'a demonstration of a person doing {{}}.',
    #             f'a demonstration of a person during {{}}.',
    #             f'a demonstration of a person performing {{}}.',
    #             f'a demonstration of a person practicing {{}}.',           
    #         ]
            
    # text_aug = [
    #             f'a video of a person {{}}.',          
    #         ]

    text_aug = [
                f'This is a video about {{}}',          
            ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        text_dict[idx] = torch.cat([clip.tokenize(template.format(c)) for i, c in data.classes])
    # classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return text_dict

def text_prompt_multi(data):
    # text_aug = ['{}']
    text_dict = {}
    text_aug = [
                f'a photo of {{}}.',
                f'a photo of a person {{}}.',
                f'a photo of a person using {{}}.',
                f'a photo of a person doing {{}}.',
                f'a photo of a person during {{}}.',
                f'a photo of a person performing {{}}.',
                f'a photo of a person practicing {{}}.',
                f'a video of {{}}.',
                f'a video of a person {{}}.',
                f'a video of a person using {{}}.',
                f'a video of a person doing {{}}.',
                f'a video of a person during {{}}.',
                f'a video of a person performing {{}}.',
                f'a video of a person practicing {{}}.',
                f'a example of {{}}.',
                f'a example of a person {{}}.',
                f'a example of a person using {{}}.',
                f'a example of a person doing {{}}.',
                f'a example of a person during {{}}.',
                f'a example of a person performing {{}}.',
                f'a example of a person practicing {{}}.',
                f'a demonstration of {{}}.',
                f'a demonstration of a person {{}}.',
                f'a demonstration of a person using {{}}.',
                f'a demonstration of a person doing {{}}.',
                f'a demonstration of a person during {{}}.',
                f'a demonstration of a person performing {{}}.',
                f'a demonstration of a person practicing {{}}.',           
            ]
            
    # text_aug = [
    #             f'a video of a person {{}}.',          
    #         ]

    # text_aug = [
    #             f'This is a video about {{}}',          
    #         ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        text_dict[idx] = torch.cat([clip.tokenize(template.format(c)) for i, c in data.classes])
    # classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return text_dict

