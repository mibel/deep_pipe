import os
import numpy as np
import nibabel as nib

from .base import Dataset


class WhiteMatterHyperintensity(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.data_path = data_path
        self.idx_to_path = self._get_idx_to_paths(data_path)
        self._patient_ids = list(self.idx_to_path.keys())

    def _get_idx_to_paths(self, path):
        idx_paths = {}
        for path_, dirs, files in os.walk(path):
            if 'pre' in dirs:
                idx = path_.split('/')[-1]
                idx_paths[idx] = path_
        return idx_paths

    def _get_quantile_img(self, img, q=95):
        a = np.zeros_like(img)
        a += np.percentile(img, q=q) / 2000
        return a

    def hist_match(self, source, template):
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        oldlen = len(source)
        nonzero_ids = []
        for i in range(len(source)):
            if source[i] > 0: nonzero_ids.append(i)

        template = template[template > 0]
        source = source[source > 0]

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        new_values = interp_t_values[bin_idx]

        rescaled_source = np.zeros(oldlen)
        j = 0
        for i in nonzero_ids:
            rescaled_source[i] = new_values[j]
            j += 1
        return rescaled_source.reshape(oldshape)


    def _reshape_to(self, tmp: np.ndarray, new_shape = None):
        """
        Reshape ND array to new shapes.
        Parameters
        ----------
        tmp : np.array
            ND array with shapes less than new_shape.
        new_shape : tuple
            Tuple from N number - new shape.
        Returns
        ------
        result : np.array
            Return np.array with shapes equal to new_shape.
         Example.
        _ = _reshape_to(X_test[..., :80], (15, 2, 300, 500, 100))
        """
        assert not new_shape is None
        new_diff = [((new_shape[-i] - tmp.shape[-i]) // 2,
                     (new_shape[-i] - tmp.shape[-i]) // 2 + \
                        (new_shape[-i] - tmp.shape[-i]) % 2)
                    for i in range(len(new_shape), 0, -1)]
        return np.pad(tmp, new_diff, mode='constant', constant_values=0)

    def load_mscan(self, patient_id):
            path_to_modalities = self.idx_to_path[patient_id]
            res = []
            for modalities in ['pre/FLAIR.nii.gz', 'pre/T1.nii.gz']: # reg_t1.nii.gz
                image = os.path.join(path_to_modalities, modalities)
                x = nib.load(image).get_data().astype('float32')

                if modalities == 'pre/FLAIR.nii.gz':
                    brain = path_to_modalities + '/pre/brainmask_T1_mask.nii.gz'
                    mask = nib.load(brain).get_data()
                    x[mask == 0] = 0

                #if 'FLAIR' in modalities:
                #    b = nib.load('/nmnt/x05-ssd/PREPR_MICCAI_WMHS/Sing/'
                #                 'Singapore/57/pre/FLAIR.nii.gz').get_data()\
                #        .astype('float32')
                #    m = nib.load(
                #        '/nmnt/x05-ssd/PREPR_MICCAI_WMHS/Sing/Singapore/57'
                #        '/pre/brainmask_T1_mask.nii.gz').get_data()\
                #        .astype('float32')
                #    b[m==0]=0
                #    x = self.hist_match(x, b)
                #    print ('matched! stripped! ')

                x = self._reshape_to(x, new_shape=self.spatial_size)

                img_std = x.std()
                x = x / img_std
                res.append(x)
            # added quantiles values
            # [res.append(self._get_quantile_img(res[0], q=q)) for q in
            #  range(0, 105, 5)]
            return np.asarray(res)

    def load_segm(self, patient_id):
        path_to_modalities = self.idx_to_path[patient_id]
        x = nib.load(os.path.join(path_to_modalities, 'wmh.nii.gz')).get_data() # 'wmh_reg.nii.gz'
        x = self._reshape_to(x, new_shape=self.spatial_size)
        x[x < 0.5] = 0
        x[np.logical_and(x >= 0.5, x < 1.5)] = 1
        x[x >= 1.5] = 0
        return np.array(x, dtype=bool)

    def load_msegm(self, patient_id):
        return self.load_segm(patient_id)[np.newaxis]

    def load_x(self, patient_id):
        return self.load_mscan(patient_id)

    def load_y(self, patient_id):
        return self.load_msegm(patient_id)

    def segm2msegm(self, segm):
        # np.array([segm == 1, segm == 2]).astype(np.int32)
        pass

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return 2

    @property
    def n_chans_msegm(self):
        return 1

    @property
    def n_classes(self):
        return 3

    @property
    def spatial_size(self):
        return (256, 256, 84)
