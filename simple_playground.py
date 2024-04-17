import math as m
import random as r
import numpy as np
from simple_geometry import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Car():
    def __init__(self) -> None:
        self.diameter = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def radius(self):
        return self.diameter/2

    # reset the car to the beginning line
    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.diameter)
        left_xpos = self.xini_min + self.diameter//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    # this is the function returning the coordinate on the right, left, front or center points
    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.diameter/2, 0).rotate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.diameter/2, 0).rotate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.diameter/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.diameter/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.diameter/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.diameter/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.diameter*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)

class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction(self, state):
        '''
        此function為模擬時，給予車子隨機數字讓其走動。
        不需使用此function。
        '''
        return r.randint(0, self.n_actions-1)

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        radius = self.car.radius

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < radius)
            p2_touch = (dp2 < radius)
            body_touch = (
                dToLine < radius and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        '''
        請更改此處code，依照自己的需求撰寫。
        '''
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state


class Animation():
    """
        一個用於在Playground中繪製動畫的類。

        Parameters:
        p (Playground): Playground對象,包含了動畫所需的所有信息。

        Attributes:
        background (list): Playground中的邊界線。
        star_line (Line): Playground中的起跑線。
        end_line (Line): Playground中的終點線。
        car_trail (list): 車子的移動軌跡。
        car_radius(int): 車子的半徑
        fig (Figure): 繪圖板的Figure對象。
        ax (Axes): 繪圖板的Axes對象。
        p (Playground): playground 的共同呼叫區域
        ani (Animation): 動畫的FuncAnimation對象。
        state (dict): Playground的當前狀態。
        """

    def __init__(self, p: Playground):
        self.background = p.lines
        self.star_line = p.decorate_lines[0]
        self.end_line = p.destination_line
        self.car_trail = []
        self.car_radius = p.car.radius
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.p = p
        self.car = None

        self.draw_background()
        self.state = p.reset()
        self.ani = FuncAnimation(self.fig, self.update, frames=np.arange(100), blit=False)

    def draw_background(self):
        """繪製Playground的背景元素。"""
        for line in self.background:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], "k-")
        self.ax.plot([self.end_line.p1.x, self.end_line.p2.x],
                     [self.end_line.p1.y, self.end_line.p2.y], "r-")
        self.ax.plot([self.star_line.p1.x, self.star_line.p2.x],
                     [self.star_line.p1.y, self.star_line.p2.y], "r-")
        self.ax.axis('equal')

    def update(self, frame):
        """更新動畫的每一幀。"""
        if not self.p.done:
            car_pos = self.p.car.getPosition("center")
            self.car_trail.append(car_pos)
            self.draw_car(car_pos)

            action = self.p.predictAction(self.state)
            self.state = self.p.step(action)

        else:
            self.ani.event_source.stop()
            print("Game over")
            # exit()

    def draw_car(self, car_pos):
        """繪製車子。"""
        self.car = plt.Circle((car_pos.x, car_pos.y), self.car_radius,
                              color="red", fill=False)
        self.ax.add_patch(self.car)


    def show_animation(self):
        """顯示動畫。"""
        plt.show()


def run_example():
    # use example, select random actions until gameover
    p = Playground()

    animation = Animation(p)
    animation.show_animation()



if __name__ == "__main__":
    run_example()
