"""Convert registration transforms and resample images in physical coordinates."""
import numpy as np
import SimpleITK as sitk

def tm_norm_to_physical(tm_norm: np.ndarray, center_y: np.ndarray, scale: float) -> np.ndarray:
    tm_norm = np.asarray(tm_norm, dtype=np.float64)
    center_y = np.asarray(center_y, dtype=np.float64).reshape(3)
    scale = float(scale)

    R = tm_norm[:3, :3]
    t_norm = tm_norm[:3, 3]
    t_phys = scale * t_norm + center_y - R @ center_y

    tm_phys = np.eye(4, dtype=np.float64)
    tm_phys[:3, :3] = R
    tm_phys[:3, 3] = t_phys
    return tm_phys

def build_sitk_affine(tm_phys: np.ndarray) -> sitk.AffineTransform:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.asarray(tm_phys[:3, :3], dtype=np.float64).reshape(-1).tolist())
    transform.SetTranslation(np.asarray(tm_phys[:3, 3], dtype=np.float64).tolist())
    return transform

def resample_moving_to_fixed(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    tm_phys: np.ndarray,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    transform = build_sitk_affine(tm_phys)
    inverse_transform = transform.GetInverse()
    return sitk.Resample(
        moving_image,
        fixed_image,
        inverse_transform,
        interpolator,
        default_value,
        moving_image.GetPixelID(),
    )

