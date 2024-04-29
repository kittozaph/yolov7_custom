import os

def draw_line(img,left,right,color=(0,0,225),line_thickness=1):

  cv2.line(img, left, right, color, thickness=line_thickness) 



def counter(founded_people,xyxy,q,top,a):
  founded_people[f"people{q}"] = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]               
  q = q + 1
  top = top + xyxy[3] - xyxy[1]
  a = True 

  return founded_people,q,top,a

def distance_counter(img,founded_people,top,q,h,plot_box,det,save_img,b,draw):
  #m = people0 #k = [10, 1.0, 12, 7.5] 
  for ş, (m, k) in enumerate(founded_people.items()):
    for o, (j, l) in enumerate(founded_people.items()):
      if not m == j:
        left_1 = np.array([k[0], k[3]])
        right_1 = np.array([k[2], k[3]])

        left_2 = np.array([l[0], l[3]])
        right_2 = np.array([l[2], l[3]])
        
        check_point1 = right_1 - left_2
        check_point2 = left_1 - right_2
        checks1 = sqrt((check_point1[0]**2) + (check_point1[1]**2))  
        checks2 = sqrt((check_point2[0]**2) + (check_point2[1]**2))

        if (checks1 < int((top)/q)/1.2573): #75
          h += 1
          plot_box (img,m,j,founded_people)
          if save_img:
            b = save(img,l,b)
          if draw:
            draw_line(img,(k[2],k[3]),(l[0],l[3]))


        if (checks2 < int((top)/q)/1.2573):
          h += 1
          plot_box (img,m,j,founded_people)
          if save_img:
            b = save(img,l,b)
          if draw:
            draw_line(img,(k[0],k[3]),(l[2],l[3]))
                    
  # cv2.putText(img,f"Close People in Pairs: = {int(h/2)} ",(0, 105), cv2.FONT_HERSHEY_TRIPLEX,1, (255, 0, 0), 1)
  cv2.putText(img,f"Total Object: = {len(det)}",(0, 255), cv2.FONT_HERSHEY_TRIPLEX,1, (200, 100, 0), 1)



def plot_box (image,m,j,founded_people,color=(0,0,225),line_thickness=3):
  tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
  c1, c2= (int(founded_people[m][0]), int(founded_people[m][3])),(int(founded_people[m][2]), int(founded_people[m][3]))
  k1, k2= (int(founded_people[j][0]), int(founded_people[j][3])),(int(founded_people[j][2]), int(founded_people[j][3]))
  cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  cv2.rectangle(image, k1, k2, color, thickness=tl, lineType=cv2.LINE_AA)

def save(im0,l,b):
  bb=im0[int(l[1]):int(l[3]),int(l[0]):int(l[2])]
  path="/content/gdrive/MyDrive/yolov7/people_imgs"
  bb_resized=cv2.resize(bb,(224,224))
  
  cv2.imwrite(os.path.join(path,f"people_close{b}.jpg"),bb_resized)
  b = b + 1
  return b

"""
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
"""

def count(classes,image):
  model_values=[]
  aligns=image.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35                                                   
    cv2.putText(image, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)