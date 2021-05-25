import cv2


def open_image_RGB(path_to_open):
    image = cv2.imread(path_to_open)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image_RGB(image, path_to_save):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_to_save, image)
