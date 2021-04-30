import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)


def image_to_blocks(image, block_width_split=3, block_height_split=4):
    image_width = image.shape[0]
    image_height = image.shape[1]
    block_width = int(image_width / block_width_split)
    block_height = int(image_height / block_height_split)
    output_row = []
    for row in range(0, image_width - block_width+1, block_width):
        output_column = []
        for column in range(0, image_height - block_height+1, block_height):
            block = image[row:row + block_width, column:column + block_height]
            resize = cv2.resize(block, (block_width * 20, block_height * 20), interpolation=cv2.INTER_AREA)
            cv2.imshow("test image", resize)
            cv2.waitKey()
            output_column.append(block)
            output_row.append(output_column)

    return output_row


def main():
    img = cv2.imread("img0.png")
    img2 = cv2.imread("IMG_0007.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    first_image_in_blocks = image_to_blocks(gray)
    sec_image_in_blocks = image_to_blocks(gray2)
    #secimageinblocks = np.array(imageToBlocks(gray2)).tolist()
    #secimageinblocks = imageToBlocks(gray2)

if __name__ == '__main__':
    main()