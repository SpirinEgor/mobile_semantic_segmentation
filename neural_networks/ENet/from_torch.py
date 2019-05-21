import numpy as np
import os
import pickle as pkl
import torchfile

CONV_TRANSPOSE = (2, 3, 1, 0)


def from_torch(torch_model):
    def expand_module(module):
        if 'weight' in module._obj or b'weight' in module._obj:
            return [module]
        if 'modules' in module._obj or b'modules' in module._obj:
            # return module._obj[b'modules']
            lst = [expand_module(submodule) for submodule in module._obj[b'modules']]
            return [sublist for item in lst for sublist in item]
        return [None]

    enet = torchfile.load(filename=torch_model)
    all_enet_modules = [module for module in expand_module(enet) if module is not None]
    all_enet_modules = [module for module in all_enet_modules if b'weight' in module._obj]

    weights = []
    # for module in all_enet_modules:
    for module in all_enet_modules:
        item = {}
        if module.torch_typename() == b'cudnn.SpatialConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        elif module.torch_typename() == b'nn.SpatialBatchNormalization':
            item = {
                'gamma': module[b'weight'],
                'beta': module[b'bias'],
                'moving_mean': module[b'running_mean'],
                'moving_variance': module[b'running_var'],
            }
        elif module.torch_typename() == b'nn.PReLU':
            weight = np.expand_dims(np.expand_dims(module[b'weight'], 0), 0)
            item['weight'] = weight
        elif module.torch_typename() == b'nn.SpatialDilatedConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        elif module.torch_typename() == b'nn.SpatialFullConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        else:
            print('Unhandled torch layer: {}'.format(module.torch_typename()))
        item['torch_typename'] = module.torch_typename().decode()

        if 'Convolution' in item['torch_typename']:
            item['weight'] = np.transpose(item['weight'], CONV_TRANSPOSE)

        weights.append(item)
    return weights

def transfer_weights(model, weights=None, keep_top=False):
    """
    Transfers weights from torch-enet if they are available as {PROJECT_ROOT}/models/pretrained/torch_enet.pkl after
    running from_torch.py.
    :param keep_top: Skips the final Transpose Convolution layer if False.
    :param model: the model to copy the weights to.
    :param weights: the filename that contains the set of layers to copy. Run from_torch.py first.
    :return: a model that contains the updated weights. This function mutates the contents of the input model as well.
    """

    def special_cases(idx):
        """
        Handles special cases due to non-matching layer sequences
        :param idx: original index of layer
        :return: the corrected index of the layer as well as the corresponding layer
        """
        idx_mapper = {
            266: 267,
            267: 268,
            268: 266,
            299: 300,
            300: 301,
            301: 299
        }

        actual_idx = idx_mapper[idx] if idx in idx_mapper else idx
        return actual_idx, model.layers[actual_idx]

    if weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.join(dir_path, os.pardir, os.pardir, os.pardir)
        weights = os.path.join(project_root, 'pretrained', 'torch_enet.pkl')
    if not os.path.isfile(weights):
        print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    else:
        weights = os.path.abspath(weights)
        print('Loading pretrained weights from {}'.format(weights))
        with open(weights, 'rb') as fin:
            weights_mem = pkl.load(fin)
            idx = 0
            for num, layer in enumerate(model.layers):
                # handle special cases due to non-matching layer sequences
                actual_num, layer = special_cases(num)

                if not layer.weights:
                    continue

                item = weights_mem[idx]
                layer_name = item['torch_typename']
                new_values = layer.get_weights()
                if layer_name in ['cudnn.SpatialConvolution',
                                  'nn.SpatialDilatedConvolution']:
                    if 'bias' in item:
                        new_values = [item['weight'], item['bias']]
                    else:
                        new_values = [item['weight']]
                elif layer_name == 'nn.SpatialBatchNormalization':
                    new_values = [item['gamma'], item['beta'],
                                  item['moving_mean'], item['moving_variance']]
                elif layer_name == 'nn.PReLU':
                    new_values = [item['weight']]
                elif layer_name == 'nn.SpatialFullConvolution':
                    if keep_top:
                        if 'bias' in item:
                            new_values = [item['weight'], item['bias']]
                        else:
                            new_values = [item['weight']]
                else:
                    print('Unhandled layer type "{}"'.format(layer_name))
                layer.set_weights(new_values)
                idx += 1
    return model
