from PIL import Image
import binascii
import io

import binascii
with open("img.jpg", "rb") as f:
    image_bytes = f.read()
hex_string = binascii.hexlify(image_bytes).decode('utf-8')
print(hex_string)

image_bytes = binascii.unhexlify(hex_string)
image = Image.open(io.BytesIO(image_bytes))

# Save or display the image
# #image.show() 
image.save("decoded_image.png")
