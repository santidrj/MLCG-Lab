from PyRT_Common import *


# -------------------------------------------------Integrator Classes
# the integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                pixel = GREEN
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        pass


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        pass


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        pass


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        pass


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
