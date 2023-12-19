import cv2

# Mouse callback function
count = 0
def save_pixel(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        with open('clicked_pixels.txt', 'a') as f:
            rgb = image[y, x]
            f.write(f'{count}: {x}, {y}, RGB: {rgb}\n')
            count += 1
        print(f'Saved pixel ({x}, {y}), RGB: {rgb}')
        cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=-1)  # BGR color
        cv2.imshow('image', image)  # Update the image display

# Load the image
image = cv2.imread('/home/lacie/Github/ComputerVisionTools/bevfusion/images-3/pointcloud_bev.png')

# Create a window
cv2.namedWindow('image')

# Set the mouse callback function
cv2.setMouseCallback('image', save_pixel)

# Display the image
cv2.imshow('image', image)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()


