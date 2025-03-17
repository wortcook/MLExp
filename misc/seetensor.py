from safetensors import safe_open

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ANI_KEY = "decoder.block.0.layer.0.SelfAttention.o.weight"
# ANI_KEY = "decoder.block.0.layer.0.SelfAttention.o.weight"
# ANI_KEY = "decoder.block.0.layer.2.DenseReluDense.wi.weight"
ANI_KEY = "decoder.block.0.layer.1.EncDecAttention.o.weight"

#loop 1 to 5

wimages = []

fig, ax = plt.subplots()

isFirst = True
isFirstAnimation = True

last_tensor = None

for i in range(1, 53):

    print(f"Loading tensor from model {i:03}")

    with safe_open(f"./{i:03}/model.safetensors", framework="pt", device="cpu") as f:
        tensor = f.get_tensor(ANI_KEY)


        #normalize the tensor to 0-255 range
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255

        if isFirst:
            last_tensor = tensor
            isFirst = False

            im = [ax.imshow(tensor.numpy(), cmap="gray")]
        else:
            diff = tensor - last_tensor

            #scale the diff to 0-255 range
            diff_min = diff.min()
            diff_max = diff.max()
            diff = (diff - diff_min) / (diff_max - diff_min) * 255

            last_tensor = tensor

            #create the image

            # if isFirstAnimation:
            #     im = [ax.imshow(diff.numpy(), cmap="gray")]
            # else:
            #     im = [ax.imshow(diff.numpy(), cmap="gray", animated=True)]
            # im = [ax.imshow(diff.numpy(), cmap="gray", animated=True)]


            #combine the images side by side
            im = [ax.imshow(np.concatenate((tensor.numpy(), diff.numpy()), axis=1), cmap="gray", animated=True)]

            # isFirstAnimation = False
            wimages.append(im)

ani = animation.ArtistAnimation(fig, wimages, interval=250, blit=True, repeat_delay=1000)

plt.show()




# tensors = {}
# with safe_open("./my_model/model.safetensors", framework="pt", device="cpu") as f:

#     for key in f.keys():
#         tensors[key] = f.get_tensor(key)

#         #if the tensor is 2d
#         if len(tensors[key].shape) == 2:

#             #scale the tensor to 0-255 range
#             tensors_min = tensors[key].min()
#             tensors_max = tensors[key].max()
#             tensors[key] = (tensors[key] - tensors_min) / (tensors_max - tensors_min) * 255

#             print(f"Tensor: {key} | Shape: {tensors[key].shape}")
#             print(f"old min: {tensors_min} | old max: {tensors_max}")
#             print(f"new min: {tensors[key].min()} | new max: {tensors[key].max()}")

#             #plot the tensor
#             plt.imshow(tensors[key].numpy(), cmap="gray")
#             plt.title(key)
#             plt.show()

#             print("*" * 20)

#             #plot as an array of images
#         else:
#             print(f"Tensor: {key} | Shape: {tensors[key].shape}")
#             print("*" * 20)
