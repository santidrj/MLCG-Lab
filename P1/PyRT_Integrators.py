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
                ray = Ray(Vector3D(0, 0, 0), cam.get_direction(x, y))
                # pixel = RGBColor(x / cam.width, y / cam.height, 0)
                pixel = self.compute_color(ray)
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
        return RED if self.scene.any_hit(ray) else BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        color = max(1 - hit.hit_distance / self.max_depth, 0)
        return RGBColor(color, color, color)


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)

        # If there is a hit we compute the color, otherwise we return black.
        # If we didn't do this the background color would be gray since we are
        # adding (1,1,1) to the zero normal vector and then dividing by 2.
        if hit.has_hit:
            if isinstance(hit.normal, Vector3D):
                c = (hit.normal + Vector3D(1, 1, 1)) / 2
                return RGBColor(c.x, c.y, c.z)
            else:
                c = (hit.normal + [1, 1, 1]) / 2
                return RGBColor(c[0], c[1], c[2])
        return BLACK


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        # light_source = self.scene.pointLights[0]

        la = self.compute_ambient_reflection(hit)
        if hit.has_hit:
            color = la
        else:
            color = BLACK
        for light_source in self.scene.pointLights:
            hit_point = hit.hit_point if isinstance(hit.hit_point, Vector3D) else Vector3D(hit.hit_point[0],
                                                                                           hit.hit_point[1],
                                                                                           hit.hit_point[2])
            direction = Normalize(hit_point - light_source.pos)
            aux_obj = self.scene.closest_hit(Ray(light_source.pos, direction))
            if hit.has_hit:
                if hit.primitive_index == aux_obj.primitive_index:
                    ld = self.compute_diffuse_reflection(hit, light_source)
                    color += ld

        return color

    def compute_ambient_reflection(self, hit):
        o = self.scene.object_list[hit.primitive_index]
        kd = o.get_BRDF().kd
        ia = self.scene.i_a
        return ia.multiply(kd)

    def compute_diffuse_reflection(self, hit, light_source):
        o = self.scene.object_list[hit.primitive_index]
        hit_point, normal = get_hit_data(hit)
        wi = Normalize(light_source.pos - hit_point)
        wo = Vector3D(0, 0, 0)
        if hit_point.x != 0 and hit_point.y != 0 and hit_point.z != 0:
            wo = Normalize(hit_point * -1)
        L = o.get_BRDF().get_value(wi, wo, normal)
        return L.multiply(light_source.intensity * hit.hit_distance ** -2)


# ############################################################################################## #
# Given a list of hemispherical functions (function_list) and a set of sample positions over the #
#  hemisphere (sample_pos_), return the corresponding sample values. Each sample value results   #
#  from evaluating the product of all the functions in function_list for a particular sample     #
#  position.                                                                                     #
# ############################################################################################## #
def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            val *= function_list[j].eval(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))  # for convenience, we'll only use the red channel
    return sample_values


def compute_estimate_cmc(sample_prob_, sample_values_):
    values = [value / prob for prob, value in zip(sample_prob_, sample_values_)]
    result = BLACK
    for value in values:
        result += value
    return result / len(sample_prob_)


def get_hit_data(hit):
    normal = hit.normal if isinstance(hit.normal, Vector3D) else Vector3D(hit.normal[0], hit.normal[1],
                                                                          hit.normal[2])
    hit_point = hit.hit_point if isinstance(hit.hit_point, Vector3D) else Vector3D(hit.hit_point[0],
                                                                                   hit.hit_point[1],
                                                                                   hit.hit_point[2])
    return hit_point, normal


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            exp = 1
            cosine_term = CosineLobe(exp)
            hit_point, normal = get_hit_data(hit)

            sample_set, sample_prob = sample_set_hemisphere(self.n_samples, UniformPDF())
            li = []
            brdf = []
            cosine = []
            for sample in sample_set:
                centered_sample = center_around_normal(sample, normal)
                second_ray = Ray(hit_point, centered_sample)

                secondary_hit = self.scene.closest_hit(second_ray)
                _, s_normal = get_hit_data(secondary_hit)
                if secondary_hit.has_hit:
                    o = self.scene.object_list[secondary_hit.primitive_index]
                    li.append(o.emission)
                else:
                    if self.scene.env_map:
                        li.append(self.scene.env_map.getValue(centered_sample))
                    else:
                        li.append(BLACK)

                o = self.scene.object_list[hit.primitive_index]
                brdf.append(o.get_BRDF().get_value(second_ray.d, ray.d, normal))
                cosine.append(cosine_term.eval(sample))

            sample_values = [l.multiply(b) * c for l, b, c in zip(li, brdf, cosine)]
            return compute_estimate_cmc(sample_prob, sample_values)

        return self.scene.env_map.getValue(ray.d)


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
