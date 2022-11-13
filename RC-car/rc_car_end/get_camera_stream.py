import curses
import time

movement_bindings = {
        curses.KEY_UP:    (1,  0),
        curses.KEY_DOWN:  (-1,  0),
        curses.KEY_LEFT:  (0,  1),
        curses.KEY_RIGHT: (0, -1),
    }

# #!/usr/bin/env python
# from __future__ import print_function

# # import roslib
# # roslib.load_manifest('my_package')
# import sys
# import rospy
# import cv2
# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError


# class image_converter:

#   def __init__(self):
#     self.image_pub = rospy.Publisher("image_topic_2",Image)

#     self.bridge = CvBridge()
#     self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)

#   def callback(self,data):
#     try:
#       cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#     except CvBridgeError as e:
#       print(e)

#     (rows,cols,channels) = cv_image.shape
#     if cols > 60 and rows > 60 :
#       cv2.circle(cv_image, (50,50), 10, 255)

#     cv2.imshow("Image window", cv_image)
#     cv2.waitKey(3)

#     try:
#       self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
#     except CvBridgeError as e:
#       print(e)

# def main(args):
#   ic = image_converter()
#   rospy.init_node('image_converter', anonymous=True)
#   try:
#     rospy.spin()
#   except KeyboardInterrupt:
#     print("Shutting down")
#   cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main(sys.argv)
# def execute(stdscr):
#     stdscr.nodelay(True)
#     curses.curs_set(0)

#     while True :
#         while True:
#             keycode = stdscr.getch()
#             if keycode == -1:
#                 break
#             if keycode in movement_bindings:
#                 print("Yesss")
#         # key = stdscr.getch()
#         # print(keycode)
#         stdscr.refresh()
#         time.sleep(1/5.)



# def main():
#     try:
#         curses.wrapper(execute)
#     except KeyboardInterrupt:
#         pass


# if __name__ == '__main__':
#     main()
