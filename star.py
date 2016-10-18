"""
Infer the local escape speed and dark matter density.
"""

import cPickle as pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pystan as stan


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # TODO: Remove this when stable.

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)


class Model(object):
    pass


class Star(Model):

    _model_path = "star.stan"

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        self._model = self._load_model(**kwargs)

        return None


    def _load_model(self, **kwargs):
        """
        Load the model, either from disk or from a pre-compiled model.
        """

        recompile = True
        compiled_path = "{}.compiled".format(self._model_path)

        if os.path.exists(compiled_path):

            with open(compiled_path, "rb") as fp:
                model = pickle.load(fp)

            # Check that the model code is the same as what we expected.
            with open(self._model_path, "r") as fp:
                model_code = fp.read()

            try:
                expected_model_code = model.model_code

            except:
                None

            else:
                if model_code == expected_model_code:
                    logger.warn(
                        "Using Pre-compiled model in {}.compiled".format(
                            self._model_path))
                    recompile = False

        if recompile:
            model = stan.StanModel(file=self._model_path)

            # Save the compiled model.
            with open(compiled_path, "wb") as fp:
                pickle.dump(model, fp, -1)

        return model


    def _validate_stan_inputs(self, **kwargs):
        """
        Check the format of the initial values for the model. If a dictionary
        is specified and multiple chains are given, then the initial value will
        be re-cast as a list of dictionaries (one per chain).
        """

        # Copy the dictionary of keywords.
        kwds = {}
        kwds.update(kwargs)

        # Allow for a keyword that will disable any verification checks.
        if not kwds.pop("validate", True):
            return kwds

        # Check chains and init values.
        if "init" in kwds.keys() and isinstance(kwds["init"], dict) \
        and kwds.get("chains", 1) > 1:

            init, chains = (kwds["init"], kwds.get("chains", 1))
            logger.info(
                "Re-specifying initial values to be list of dictionaries, "\
                "allowing one dictionary per chain ({}). "\
                "Specify validate=False to disable this behaviour"\
                .format(chains))
            
            kwds["init"] = [init] * chains

        return kwds


    def optimize(self, data, **kwargs):
        """
        Optimize the model given the data. Keyword arguments are passed directly
        to the `StanModel.optimizing` method.

        :param data:
            A dictionary containing the required key/value pairs for the STAN
            model.
        """

        kwds = self._validate_stan_inputs(data=data, **kwargs)
        return self._model.optimizing(**kwds)


    def sample(self, data, chains=4, iter=2000, warmup=None, **kwargs):
        """
        Draw samples from the model. Keyword arguments are passed directly to
        `StanModel.sampling`.

        :param data:
            A dictionary containing the required key/value pairs for the Stan
            model.

        :param chains: [optional]
            Positive integer specifying the number of chains.

        :param iter: [optional]
            Positive integer specifying how many iterations for each chain
            including warmup.

        :param warmup: [optional]
            Positive integer specifying the number of warmup (aka burn-in)
            iterations. As warm-up also specifies the number of iterations used
            for step-size adaption, warmup samples should not be used for
            inference. Defaults to iter // 2.
        """

        kwds = self._validate_stan_inputs(
            data=data, chains=chains, iter=iter, warmup=warmup, **kwargs)
        return self._model.sampling(**kwds)



if __name__ == '__main__':

    from astropy.table import Table

    overwrite = True

    model = Star()
    tgas = Table.read("/Users/arc/research/gaia/stacked_tgas.fits")

    N = len(tgas)
    indices = np.random.choice(range(N), N, replace=False)

    results = []

    for i, index in enumerate(indices):

        row = tgas[index]

        print("At star {}/{}: {}".format(i + 1, N, row["source_id"]))

        result_file = "{source_id}.pkl".format(source_id=row["source_id"])

        if os.path.exists(result_file) and not overwrite:
            print("Skipping {} because it already exists: {}".format(
                row["source_id"], result_file))
            continue


        # Build the requisite data array and covariance matrix
        y = np.array([row["parallax"], row["pmra"], row["dec"]])
        Sigma = np.eye(3) * np.array([
            np.sqrt(row["parallax_error"]**2 + 0.09), # 0.09 = 0.3**2
            row["pmra_error"],
            row["pmdec_error"]
        ])**2

        params = ("parallax", "pmra", "pmdec")
        for i, pi in enumerate(params):
            for j, pj in enumerate(params):
                if i <= j: continue
                Sigma[i, j] = Sigma[j, i] = row["{}_{}_corr".format(pj, pi)] \
                    * np.sqrt(Sigma[i, i] * Sigma[j, j])

        data = {
            "L": 1.35, # [kpc]
            "y": y,
            "Sigma": Sigma,
            "solar_motion": 0.0 # WRONG #TODO #HACK
        }

        op_params = model.optimize(data=data)

        sampled = model.sample(data=data, init=op_params)

        # Save all information from the model.
        result = dict(
            data=data,
            op_params=op_params,
            samples=sampled.extract(), 
            summary="{}".format(sampled),
            model_code=sampled.stanmodel.model_code)

        with open(result_file, "wb") as fp:
            pickle.dump(result, fp, -1)

        p = np.percentile(result["samples"]["d"], [16, 50, 84])
        central, pos, neg = (p[1], p[2] - p[1], p[0] - p[1])

        print(result["summary"])
        results.append([row["source_id"], central, pos, neg])



