import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class ChengPrc05(WhiteBalanceAlgorithm):
    """
    Implementation of Cheng et al. "Illuminant Estimation for Color Constancy: 
    why spatial domain methods work and the role of the color distribution" (JOSA 2014)
    with fixed prc = 0.005
    """
    def __init__(self):
        super().__init__()
        self.prc = 0.005
    
    def _estimate(self, data, process_masked=False):
        image = data.get_raw_image()  # Normalized image [0, 1]
        # Get flattened pixels (H*W, 3) and remove masked pixels 
        pixels = self._get_pixels(image, data, process_masked)
        
        mask = np.all(pixels < 0.98, axis=1) # Threshold that nus8 camera possibly saturates at 0.98, taken from the paper
        pixels = pixels[mask]

        if pixels.shape[0] == 0:
            return {
                "single_illuminant": (1.0, 1.0),
                "multi_illuminants": None,
                "illuminant_map": None,
                "estimated_srgb_image": None
            }
        
        # Mean for Gray World estimation
        l = np.mean(pixels, axis=0)
        norm_l = np.linalg.norm(l)
        if norm_l > 0:
            l = l / norm_l
        else:
            l = np.array([1, 1, 1]) / np.sqrt(3)

        # Projection onto Gray World estimation
        data_p = pixels @ l
        
        # Sort indices based on projections
        idx = np.argsort(data_p)
        n = pixels.shape[0]
        
        # Select both ends of the distribution
        n_sel = int(np.ceil(n * self.prc))
        n_bottom = int(np.floor(n * (1.0 - self.prc)))
        data_selected = np.vstack((pixels[idx[:n_sel]], pixels[idx[n_bottom-1:]]))
        
        if data_selected.shape[0] == 0:
            # Fallback (should not happen normally)
            r_g, b_g = l[2]/l[1], l[0]/l[1]
            return {
                "single_illuminant": (r_g, b_g),
                "multi_illuminants": None,
                "illuminant_map": None,
                "estimated_srgb_image": None
            }

        # PCA - Eigenvalue decomposition of autocorrelation matrix
        sigma = data_selected.T @ data_selected
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        
        # Largest eigenvalue corresponding eigenvector
        # eigh returns eigenvalues in ascending order
        v = eigenvectors[:, -1]
        
        ei = np.abs(v)  # Absolute values as per MATLAB code: ei = abs(v(:,3))
        
        # We need (R/G, B/G)
        b_est, g_est, r_est = ei
        if g_est == 0:
            g_est = 1e-6
            
        r_g = r_est / g_est
        b_g = b_est / g_est
        
        return {
            "single_illuminant": (r_g, b_g),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None
        }
