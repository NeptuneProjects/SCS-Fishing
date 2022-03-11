 #!/usr/bin/env python3

"""
By William Jenkins
wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography
University of California San Diego
February 2022

South China Sea Fishing Data Analysis Project
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import operator
from pathlib import Path
import pickle
import warnings

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class DataAnalyzer(object):
    
    def __init__(self, points, areas, path):
        self.points = points
        self.areas = areas
        self.path = path
        self.dti, self.n_ships, self.n_hours, self.n_fishinghours = \
            self.load_arrays(path)
        self.df = self.load_csv_to_df(points, path)

    
    def parse_dataset(self, dataset, start=None, end=None):
        
        data = getattr(self, dataset)
        idx = self._get_index(self.dti, start, end)
        data = np.squeeze(data[idx])

        if dataset == "n_ships":
            cbar_label = "Mean Ships per Day"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                data = np.mean(data, axis=0)
        elif dataset == "n_hours":
            cbar_label = "Total Ship-Hours"
            data = np.sum(data, axis=0)
        elif dataset == "n_fishinghours":
            cbar_label = "Total Fishing Hours"
            data = np.sum(data, axis=0)
        return data, cbar_label


    def plot_histogram(
            self,
            selfmask=True,
            start=None,
            end=None,
            dataset="n_fishinghours",
            savepath=None,
            **kwargs
        ):

        data, cbar_label = self.parse_dataset(dataset, start, end)

        if selfmask:
            data = self.selfmask(data)

        fig = self._heatmap(
            data,
            cbar_label=cbar_label,
            xticklabels=self.points.countries,
            yticklabels=self.areas.countries,
            **kwargs
        )

        if savepath is not None:
            self.savepath = Path(savepath)
            fig.savefig(
                self.savepath,
                dpi=300,
                facecolor="white",
                bbox_inches="tight"
            )
        
        return fig


    @staticmethod
    def _heatmap(
            data,
            cbar_label=None,
            xticklabels=None,
            yticklabels=None,
            figsize=(4,3),
            title=None,
            show=True
        ):
        cbar_kws = {"label": cbar_label}

        fig = plt.figure(figsize=figsize, facecolor='white')
        
        sns.heatmap(
            data,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cmap=sns.color_palette("Blues", as_cmap=True),
            cbar_kws=cbar_kws
        )
        plt.xlabel("Vessel Flag")
        plt.ylabel("EEZ")
        if title is not None:
            plt.title(title)

        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


    @staticmethod
    def _get_index(dti, start, end):

        if start is not None:
            start = pd.Timestamp(start)
            crit1 = (dti >= start)
        else:
            crit1 = np.ones((len(dti),), dtype=bool)

        if end is not None:
            end = pd.Timestamp(end)
            crit2 = (dti < end)
        else:
            crit2 = np.ones((len(dti),), dtype=bool)

        return np.argwhere(crit1 & crit2)
    

    @staticmethod
    def load_arrays(path):
        data = np.load(f"{path}.npz")
        return data["index"], data["n_ships"], data["n_hours"], data["n_fishinghours"]
    

    @staticmethod
    def load_csv_to_df(points, path):
        df = pd.read_csv(
            f"{path}.csv",
            index_col=[0, 1],
            header=[0, 1],
            skipinitialspace=True,
            parse_dates=True
        )
        # Filter data from countries not specified
        for country in df.columns.levels[1]:
            if country not in points.countries:
                df = df.drop(country, axis=1, level=1)
                df = df.drop(country, axis=0, level=1)
        
        # Filter data outside dates of interest
        criteria = (df.index.get_level_values(0) >= points.start) & \
            (df.index.get_level_values(0) < points.end)
        
        return df[criteria]
    

    @staticmethod
    def selfmask(data):
        if len(data.shape) == 2:
            diag_idx = np.diag_indices_from(data)
            data[diag_idx] = np.NaN
        elif len(data.shape) == 3:
            diag_idx = np.diag_indices_from(data[0])
            for idx in range(data.shape[0]):
                data[idx][diag_idx] = np.NaN
        return data


class DataProcessor(object):

    def __init__(self, points, areas):
        self.points = points
        self.areas = areas
        self.countries = points.countries
        self.extent = points.extent
        self.crs = points.crs


    def construct_results_df(self, index, results):
        data0 = results[0].reshape(-1, results[0].shape[1])
        data1 = results[1].reshape(-1, results[1].shape[1])
        data2 = results[2].reshape(-1, results[2].shape[1])

        midx_rows = pd.MultiIndex.from_product(
            [pd.to_datetime(index), self.countries],
            names=["Date", "EEZ"]
        )

        df0 = pd.DataFrame(data0, index=midx_rows)
        df1 = pd.DataFrame(data1, index=midx_rows)
        df2 = pd.DataFrame(data2, index=midx_rows)

        df = pd.concat([df0, df1, df2], axis=1)
        df.columns = pd.MultiIndex.from_product(
            [
                [
                    "Number of Ships",
                    "Number of Ship-Hours",
                    "Number of Fishing Hours"
                ],
                self.countries]
        )
        return df


    def load(self, fname, countries=None, extent=None, crs=None):
        df = self.load_csv_to_df(fname, countries=countries, extent=extent)
        return self.format_df_to_gdf(df, crs=crs)


    def pipeline(self, f, **kwargs):
        gdf = self.load(f, **kwargs)
        return self.count(gdf, self.areas.gdf, self.countries)


    def process(
            self,
            countries=None,
            extent=None,
            crs=None,
            savepath=None,
            num_workers=1
        ):
        if countries is not None:
            self.countries = countries
        if extent is not None:
            self.extent = extent
        if crs is not None:
            self.crs = crs
        if savepath is not None:
            self.savepath = savepath

        pbargs = {
                'total': len(self.points.flist),
                'desc': 'Processing',
                'unit': 'files',
                'bar_format': '{l_bar}{bar:20}{r_bar}{bar:-20b}',
                'leave': True
            }

        kwargs = {
            "countries": self.countries,
            "extent": self.extent,
            "crs": self.crs
        }

        counts = (
            np.zeros(
                (
                    len(self.points.flist),
                    len(self.countries),
                    len(self.countries)
                )
            ),
            np.zeros(
                (
                    len(self.points.flist),
                    len(self.countries),
                    len(self.countries)
                )
            ),
            np.zeros(
                (
                    len(self.points.flist),
                    len(self.countries),
                    len(self.countries)
                )
            )
        )

        if num_workers == 1:
            for i, f in tqdm(enumerate(self.points.flist), **pbargs):
                (counts[0][i], counts[1][i], counts[2][i]) = self.pipeline(
                    f, **kwargs
                )
        
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i, f in enumerate(self.points.flist):
                    future = executor.submit(self.pipeline, f, **kwargs)
                    futures.append(future)

                for i, future in tqdm(enumerate(futures), **pbargs):
                    (counts[0][i], counts[1][i], counts[2][i]) = future.result()
        
        self.results_df = self.construct_results_df(self.points.dti, counts)

        if hasattr(self, "savepath"):
            self.save_results(
                self.savepath,
                self.points.dti,
                counts,
                self.results_df)

        return self.points.dti, counts


    @staticmethod
    def count(points, area, countries):
        n_ships = np.zeros((len(countries), len(countries)))
        n_hours = np.zeros((len(countries), len(countries)))
        n_fishinghours = np.zeros((len(countries), len(countries)))

        for i, country_i in enumerate(countries):
            filter_eez = area["ISO_SOV1"] == country_i
            subarea = area[filter_eez]

            for j, country_j in enumerate(countries):
                filter_flag = points["flag"] == country_j
                ships_by_flag = points[filter_flag]
                ships_in_area = gpd.sjoin(
                    ships_by_flag,
                    subarea,
                    predicate="within"
                )

                n_ships[i, j] = ships_in_area["mmsi_present"].sum()
                n_hours[i, j] = ships_in_area["fishing_hours"].sum()
                n_fishinghours[i, j] = ships_in_area["hours"].sum()

        return n_ships, n_hours, n_fishinghours


    @staticmethod
    def format_df_to_gdf(df, crs=None):
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df.cell_ll_lon,
                df.cell_ll_lat
            ),
            crs=crs
        )
        return gdf


    @staticmethod
    def load_csv_to_df(fname, countries=None, extent=None):
        df = pd.read_csv(fname, header=0)
        # Remove activity with insufficient data
        df = df.dropna()
        # Remove activity not associated with countries of interest
        if countries is not None:
            df = df.drop(df[~df["flag"].isin(countries)].index)
        # Remove activity outside geographic constraints
        if extent is not None:
            df = df.drop(
                df[
                    (df["cell_ll_lon"] <= extent[0]) | \
                    (df["cell_ll_lat"] <= extent[1]) | \
                    (df["cell_ll_lon"] >= extent[2]) | \
                    (df["cell_ll_lat"] >= extent[3])
                ].index
            )
        return df


    @staticmethod
    def save_results(path, index, results, df):
        np.savez(
            path,
            index=index,
            n_ships=results[0],
            n_hours=results[1],
            n_fishinghours=results[2]
        )
        df.to_csv(f"{path}.csv")


class FishingData(object):

    def __init__(
            self,
            path,
            start=None,
            end=None,
            countries=None,
            extent=None,
            crs=None
        ):
        self.path = path
        self.start = start
        self.end = end
        self.countries = countries
        self.extent = extent
        self.crs = crs


    def set_dates(self, path=None, start=None, end=None):

        if path is not None:
            self.path = path
        if start is not None:
            self.start = start
        if end is not None:
            self.end = end

        flist, dti = self.__get_files_and_dates(self.path)
        self.flist, self.dti = self.__trim_files_and_dates(
            flist,
            dti,
            start=self.start,
            end=self.end
        )


    @staticmethod
    def __get_files_and_dates(path):
        pathgen = path.glob("**/**/*.csv")
        filelist = [f for f in pathgen if f.is_file()]
        dti = pd.to_datetime([f.name[:-4] for f in filelist], format="%Y-%m-%d")

        enum_obj = enumerate(dti)
        sorted_pairs = sorted(enum_obj, key=operator.itemgetter(1))
        sorted_indices = [index for index, _ in sorted_pairs]

        flist = [filelist[index] for index in sorted_indices]
        dti = pd.to_datetime([dti[index] for index in sorted_indices])

        return flist, dti


    @staticmethod
    def __trim_files_and_dates(flist, dti, start=None, end=None):
        
        if start is not None:
            start = pd.Timestamp(start)
            crit1 = (dti >= start)
        else:
            crit1 = np.ones((len(flist),), dtype=bool)

        if end is not None:
            end = pd.Timestamp(end)
            crit2 = (dti < end)
        else:
            crit2 = np.ones((len(flist),), dtype=bool)

        criteria = crit1 & crit2
        flist = [f for i, f in enumerate(flist) if criteria[i]]
        dti = dti[criteria]
        
        return flist, dti


class GeographicData(object):

    def __init__(self, paths, countries=None, extent=None):
        self.paths = paths
        self.countries = countries
        self.extent = extent

    
    class Borders(object):

        def __init__(self, path):
            self.path = path
            self.gdf = self.load_data()

        
        def load_data(self, path=None):
            if path is not None:
                self.path = path
            return gpd.read_file(self.path)

    
    class EEZ(object):

        def __init__(self, path, countries=None):
            self.path = path
            self.countries = countries
            self.gdf = self.load_data()
        
            
        def load_data(self, path=None, countries=None):
            if path is not None:
                self.path = path
            if countries is not None:
                self.countries = countries
            eez = gpd.read_file(self.path)
            criteria = eez["POL_TYPE"] == "200NM"            
            if self.countries is not None:
                criteria = (criteria) & (eez["ISO_SOV1"].isin(self.countries))
            return eez[criteria]
    

    class NineDash(object):

        def __init__(self, path):
            self.path = path
            self.gdf = self.load_data()
        

        def load_data(self, path=None):
            if path is not None:
                self.path = path
            return gpd.read_file(self.path)
    

    def load_all(self):
        self.borders = self.load_borders()
        self.eez = self.load_eez()
        self.ninedash = self.load_ninedash()
        self.crs = self.eez.gdf.crs


    def load_borders(self):
        return self.Borders(path=self.paths["path_borders"])

    
    def load_eez(self):
        return self.EEZ(path=self.paths["path_eez"], countries=self.countries)
    

    def load_ninedash(self):
        return self.NineDash(path=self.paths["path_ninedash"])
    

    def make_scs_map(self, show=True, savepath=None):        
        extent = self.extent_formatter(self.extent, to="mpl")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis(extent)
        # Plot EEZ
        self.eez.gdf.plot(ax=ax, ec="#000000", fc="#00008B25", linewidth=0.75)
        # Plot National Borders
        self.borders.gdf.geometry.boundary.plot(
            ax=ax,
            edgecolor="black",
            color=None,
            linewidth=0.75
        )
        # Plot Nine-Dash Line
        self.ninedash.gdf.plot(ax=ax, color="red", linewidth=2)
        # Add Background Basemap
        cx.add_basemap(
            ax=ax,
            source=self.paths["path_raster"],
            crs=self.eez.gdf.crs
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(
            "South China Sea: Notional EEZ Delineations",
            fontweight="bold",
            size=14
        )

        if show:
            plt.show()
        else:
            plt.close()
        if savepath is not None:
            self.savepath = Path(savepath)
            fig.savefig(
                self.savepath,
                dpi=300,
                facecolor="white",
                bbox_inches="tight"
            )

        return fig
    

    @staticmethod
    def extent_formatter(extent, to="mpl"):
            extent = extent[0], extent[2], extent[1], extent[3]
            if to == "mpl":
                return list(extent)
            else:
                return tuple(extent)


def load_config(path):
    with open(path, "rb") as h:
        return pickle.load(h)


def save_config(path, **kwargs):
    config = {key: value for key, value in kwargs.items()}
    with open(f"{path}.pkl", "wb") as h:
        pickle.dump(config, h)
    with open(f"{path}.txt", "w") as h:
        h.write(str(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to read in and calculate basic " + \
            "statistics for fishing vessel activity data from the " + \
            "South China Sea collected by Global Fishing Watch " + \
            "(https://globalfishingwatch.org). Results are saved " + \
            "to files for later analysis."
    )
    parser.add_argument("savename", type=str, help="Enter name of outputs.")
    parser.add_argument("--path", type=str, help="Enter path to saved files.")
    parser.add_argument("--start", type=str, help="Format: YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Format: YYYY-MM-DD")
    parser.add_argument("--nw", default=1, type=int, help="Number of workers")
    args = parser.parse_args()

    # Define countries of interest
    countries = [
        "CHN", # China
        "TWN", # Taiwan
        "MYS", # Malaysia
        "PHL", # Philippines
        "BRN", # Brunei
        "VNM", # Vietnam
        "IDN", # Indonesia
        "SGP", # Singapore
        "HKG", # Hong Kong
        "KHM", # Cambodia
        "THA"  # Thailand
    ]

    # Define extent of geographic interest
    min_longitude = 99
    max_longitude = 122
    min_latitude = -3
    max_latitude = 25
    extent = (min_longitude, min_latitude, max_longitude, max_latitude)

    # Set path to data
    datapath = Path("../../Data")

    # Set path to fishing activity data
    path_fishing = datapath / "GFW" / "fleet-daily"
    
    # Set paths to geographic features
    mapdatapath = datapath / "Mapping"
    path_borders = mapdatapath / \
        "ne_10m_admin_0_countries" / \
            "ne_10m_admin_0_countries.shp"
    path_eez = mapdatapath / \
        "World_EEZ_v11_20191118_gpkg" / \
            "eez_v11.gpkg"
    path_ninedash = mapdatapath / \
        "MaritimeClaim_China9DashLines" / \
            "MaritimeClaim_China9DashLines.gpkg"
    path_raster = mapdatapath / \
        "NE1_HR_LC_SR_W_DR" / \
            "NE1_HR_LC_SR_W_DR.tif"
    mappaths = {
        "path_borders": path_borders,
        "path_eez": path_eez,
        "path_ninedash": path_ninedash,
        "path_raster": path_raster,
    }

    # Set path and names of saved files
    if args.path is not None:
        savepath = Path(args.path)
    else:
        savepath = Path.cwd().parent.parent / "Data" / "Processed"
    savepath = savepath / args.savename

    # Save script configuration to file for later analysis
    save_config(
        savepath,
        countries=countries,
        extent=extent,
        paths=mappaths,
        path_fishing=path_fishing,
        savepath=savepath,
        start=args.start,
        end=args.end
    )

    # Instantiate object containing geographical data
    geodata = GeographicData(paths=mappaths, countries=countries, extent=extent)
    geodata.load_all()
    EEZ = geodata.eez

    # Instantiate object containing fishing activity data
    activity = FishingData(
        path_fishing,
        countries=countries,
        extent=extent,
        crs=EEZ.gdf.crs
    )
    activity.set_dates(start=args.start, end=args.end)

    # Instantiate data processing object and process data
    dp = DataProcessor(activity, EEZ)
    _, _ = dp.process(savepath=savepath, num_workers=args.nw)