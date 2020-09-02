import cv2
import numpy as np


def count_above_thres(cnt_list, thres):
    cnt_array = []
    for x in cnt_list:
        if x>thres:
            cnt_array.append(1)
    return sum(cnt_array)
def swap_image_colors(img, low_color, high_color):
    swap_img = img.copy()
    swap_img[swap_img==low_color] = 1000
    swap_img[swap_img==high_color] = low_color
    swap_img[swap_img==1000] = high_color    
    return swap_img 

def img_bg_detector_hor(input_img, high_color):    
    img = input_img.copy()
    img[img == high_color] = 1
    max_cnts_list = []
    for rr in range(img.shape[0]):       
        max_cnts = [0,0]
        cnts = [0,0]    
        
        prev = img[rr,0]
        
        for cc in range(1,img.shape[1]):
            pix = img[rr,cc]
            
            
            if (pix != prev):
                if (max_cnts[prev]<cnts[prev]):
                    max_cnts[prev] = cnts[prev]
                cnts[prev] = 0
            #print(f'cnts pix : {cnts, pix}')    
            cnts[pix]= cnts[pix]+1
            #print([prev, pix, cnts, max_cnts])
            prev = pix
        
        if (max_cnts[prev]<cnts[prev]):
            max_cnts[prev] = cnts[prev]   
        #print('End of line')
        #print([cnts, max_cnts])    
        max_cnts_list.append(max_cnts)
        
    return max_cnts_list

def img_bg_detector_ver(input_img, high_color):    
    img = input_img.copy()
    img[img == high_color] = 1

    max_cnts_list = []
    for cc in range(img.shape[1]):       
        max_cnts = [0,0]
        cnts = [0,0]    
        
        prev = img[0,cc]
        
        for rr in range(1,img.shape[0]):
            
            pix = img[rr,cc]
            
            if (pix != prev):
                if (max_cnts[prev]<cnts[prev]):
                    max_cnts[prev] = cnts[prev]
                cnts[prev] = 0
                
            cnts[pix]= cnts[pix]+1
            #print([rr,cc ,prev, pix, cnts, max_cnts])
            prev = pix
        
        if (max_cnts[prev]<cnts[prev]):
            max_cnts[prev] = cnts[prev]   
        #print('End of line')
        #print([cnts, max_cnts])    
        max_cnts_list.append(max_cnts)
        #print(max_cnts_list)
        
    return max_cnts_list

def convert_binary_to_black_on_white(binary):
    thres = 0.8
    low_color = 0
    high_color = 255

    [w,h] = binary.shape
    #print(w,h)
    max_list_hor = img_bg_detector_hor(binary, high_color)

    cnt_low_list_hor = [xy[0]/w for xy in max_list_hor]
    cnt_high_list_hor = [xy[1]/w for xy in max_list_hor]

    #plt.figure(1)
    #plt.plot(cnt_low_list_hor)
    #plt.plot(cnt_high_list_hor)
    #plt.legend(['Low','High'])

    #[h,w] = binary_resized.shape

    max_list_ver = img_bg_detector_ver(binary, high_color)

    cnt_low_list_ver = [xy[0]/h for xy in max_list_ver]
    cnt_high_list_ver = [xy[1]/h for xy in max_list_ver]

    #plt.figure(2)
    #plt.plot(cnt_low_list_ver)
    #plt.plot(cnt_high_list_ver)
    #plt.legend(['Low','High'])


    low_cnt = count_above_thres(cnt_low_list_hor, thres) + count_above_thres(cnt_low_list_ver, thres)
    high_cnt = count_above_thres(cnt_high_list_hor, thres) + count_above_thres(cnt_high_list_ver, thres)

    #print([low_cnt,high_cnt])

    if (low_cnt > high_cnt):
        black_on_white_binary = swap_image_colors(binary, low_color, high_color)
    else:
        black_on_white_binary = binary
        
    return black_on_white_binary


def convert_binary_to_white_on_black(binary):
    thres = 0.8
    low_color = 0
    high_color = 255

    [w,h] = binary.shape
    #print(w,h)
    max_list_hor = img_bg_detector_hor(binary, high_color)

    cnt_low_list_hor = [xy[0]/w for xy in max_list_hor]
    cnt_high_list_hor = [xy[1]/w for xy in max_list_hor]

    #plt.figure(1)
    #plt.plot(cnt_low_list_hor)
    #plt.plot(cnt_high_list_hor)
    #plt.legend(['Low','High'])

    #[h,w] = binary_resized.shape

    max_list_ver = img_bg_detector_ver(binary, high_color)

    cnt_low_list_ver = [xy[0]/h for xy in max_list_ver]
    cnt_high_list_ver = [xy[1]/h for xy in max_list_ver]

    #plt.figure(2)
    #plt.plot(cnt_low_list_ver)
    #plt.plot(cnt_high_list_ver)
    #plt.legend(['Low','High'])


    low_cnt = count_above_thres(cnt_low_list_hor, thres) + count_above_thres(cnt_low_list_ver, thres)
    high_cnt = count_above_thres(cnt_high_list_hor, thres) + count_above_thres(cnt_high_list_ver, thres)

    #print([low_cnt,high_cnt])

    if (low_cnt < high_cnt):
        black_on_white_binary = swap_image_colors(binary, low_color, high_color)
    else:
        black_on_white_binary = binary
        
    return black_on_white_binary


def get_image_formats(bgr_img):

    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(bgr_img, alpha=(255.0))

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(7,7),0)

    # Applied inversed thresh_binary 
    binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary_inverted = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    
    binary = dilate
    
    black_on_white_binary = convert_binary_to_black_on_white(binary)
    white_on_black_binary = convert_binary_to_white_on_black(binary)
    
    all_image_formats =[]
    all_image_formats.append(plate_image.copy())
    all_image_formats.append(gray.copy())
    all_image_formats.append(binary.copy())
    all_image_formats.append(binary_inverted.copy())
    all_image_formats.append(black_on_white_binary.copy())
    all_image_formats.append(white_on_black_binary.copy())
    
    
    return all_image_formats


# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


def get_countours(plate_image, binary):
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    #print(len(cont))
    
    if(len(cont)>0):
        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                    # Sperate number and gibe prediction
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

    #print("Detect {} letters...".format(len(crop_characters)))
    #fig = plt.figure(figsize=(10,6))
    #plt.axis(False)
    #plt.imshow(test_roi)
    
    return crop_characters, test_roi
    #plt.savefig('grab_digit_contour.png',dpi=300)
                
def sort_contours(cnts,reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts
    
