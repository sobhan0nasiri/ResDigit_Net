import cv2
import numpy as np

def preprocess_image_for_mnist(image_path, target_size=28):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Not found!")
    
    img = cv2.bitwise_not(img)

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        
        print("Not found digit contours!")
        return np.zeros((target_size, target_size), dtype=np.uint8)

    c = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(c)
    
    digit_roi = thresh[y:y+h, x:x+w]
    
    rows, cols = digit_roi.shape
    
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
    
    digit_roi = cv2.resize(digit_roi, (cols, rows), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((target_size, target_size), dtype=np.uint8)
    
    pad_top = (target_size - rows) // 2
    pad_left = (target_size - cols) // 2
    
    final_img[pad_top:pad_top+rows, pad_left:pad_left+cols] = digit_roi

    final_img = final_img.astype('float32') / 255.0
    
    final_img = np.expand_dims(final_img, axis=-1)

    return final_img

if __name__ == "__main__":
    
    input_image = "D:\Programing Project\Project\ResDigit Net\digits-backend\history\image1.png" 
    
    try:
        processed_img = preprocess_image_for_mnist(input_image)

        print(f"Shape of processed image: {processed_img.shape}")
        
        window_name = "Final Input to Model"
        cv2.imshow(window_name, (processed_img * 255).astype(np.uint8))
        
        print("Image displayed successfully.")
        
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")