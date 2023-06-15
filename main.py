import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os



st.title("Image Uploader")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    root_path = os.getcwd()+"/"+uploaded_file.name
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # imgname = "/content/img2_m3.jpeg"
    gray = cv2.imread(root_path,0)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, threshed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY )
    h,w = img.shape[:2]
    x = np.sum(threshed, axis=0)
    y = np.sum(threshed, axis=1)
    yy = np.nonzero(y>(w/5*255))[0]
    xx = np.nonzero(x > (h/5*255))[0]
    region = img[yy[0]:yy[-1], xx[0]:xx[-1]]
    # cv2.imwrite("region1.png", region)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    imglab = np.hstack((l,a,b))
    # cv2.imwrite("region_lab.png", imglab)
    na = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # cv2.imwrite("region_a_normalized.png", na)


    retval, threshed = cv2.threshold(na, thresh = 180,  maxval=255, type=cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE , (3,3))
    opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN,kernel)
    res = np.hstack((threshed, opened))

    contours = cv2.findContours(opened, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
    res = region.copy()
    cv2.drawContours(res, contours, -1, (255,0,0), 1)
    results_nd = {}
    for idx, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        area = bbox[-1]*bbox[-2]
        if area < 100:
            continue
        rot_rect = cv2.minAreaRect(contour)
        (cx,cy), (w,h), rot_angle = rot_rect
        rbox = np.int0(cv2.boxPoints(rot_rect))
        cv2.drawContours(res, [rbox], 0, (0,255,0), 1)
        text="#{}: {:2.3f}".format(idx, rot_angle)
        results_nd[idx] = round(float(rot_angle),2)
        org=(int(cx)-10,int(cy)-10)
        cv2.putText(res, text=text, org = org, fontFace = 1, fontScale=0.8, color=(0,0,255), thickness = 1, lineType=16)
    
    print("Done")

    st.image(res, caption="captured Readings", use_column_width=True)
    print(results_nd)
    st.header("Meter Readings")
    for k, v in results_nd.items():
        st.write(k,v)
    


    cv2.imwrite("img2_m3_readings.png", res)
