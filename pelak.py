import cv2
import numpy as np


class Pelak:
    def __init__(self, path):
        self.path = path
        self.read_pelak()

    def read_pelak(self):
        self.pelak = cv2.imread(self.path)
        pelakgray = cv2.cvtColor(self.pelak, cv2.COLOR_BGR2GRAY)
        pelakblur = cv2.blur(pelakgray, (5, 5))
        _,self.pelakbw = cv2.threshold(pelakblur, 40, 255, 0)
        cv2.imwrite("pelakbw.jpg",self.pelakbw)

    def crop(self, x0, x1):
        single_number = self.pelakbw[:, x0:x1]
        single_number_resized = cv2.resize(single_number, (8, 32))
       
        single_number_flattened = single_number_resized.flatten()
      
        return single_number_flattened

    def segments(self):
        ALL_WHITES = 90*255
        segments = []    
        j = 0
        for i in range(411) :
            if np.sum(self.pelakbw[:, i]) != ALL_WHITES : #the numbers
                continue
            if j == 0 :
                x0 = i
                j += 1
                continue
            if j == 1 :
                x1 = i
                if x1 - x0 > 10 :
                    segments.append(self.crop(x0, x1))
                    j = 0
                else :
                    x0 = x1
        
        return np.array(segments)
    
    def pelak_show(self, img):
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write(self, acc, predictions):
        result = self.pelak.copy()
        result = cv2.putText(result, f"{predictions[0]} {predictions[1]}      {predictions[3]}  {predictions[4]} {predictions[5]}   {predictions[6]} {predictions[7]}", (20, 25) ,cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        result = cv2.putText(result, f"accuracy = {acc: .2f}", (20, 60) ,cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        cv2.imwrite("./result.jpg", result)
        self.pelak_show(result)

    