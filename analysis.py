import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

longitude = 32.5657  # longitude in degrees
latitude = 0.3370   # latitude in degrees
surtilt = 30  # surface tilt in degrees
surazim = 0  # surface azimuth in degrees
sc = 1367  # solar constant


def to_doy(datetime):
    """this function recieves pandas datetime index with GMT timezone aware
    returns: day of year"""
    try:
        dt = datetime.tz_convert("Africa/Kampala")  # timezone of the region your in
        nor_yr = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        lep_yr = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        if dt.month == 1:
            doy = dt.day
        elif dt.month != 1 and dt.year % 4 != 0:
            doy = dt.day + nor_yr[:dt.month - 1].sum()
        else:
            doy = dt.day + lep_yr[:dt.month - 1].sum()

        return doy
    except Exception as e:
        return e


def declination_cooper69(datetime):
    """this functions finds
    solar declination in degrees using cooper formular
    returns: declination """
    try:
        DOY = to_doy(datetime)
        radians = np.radians((360 / 365) * (DOY + 284))
        declination = 23.45 * np.sin(radians)
        return declination
    except Exception as e:
        return e


def day_angle(date_time):
    """This function calculates simple day angle in degrees
     returns: day angle"""
    try:
        DOY = to_doy(date_time)
        return (360 / 365) * (DOY - 1)
    except Exception as e:
        return e


def equation_of_time_pvcdrom(datetime):
    """this function computes equation of time
    returns: time in minutes"""
    try:
        DOY = to_doy(datetime)
        B_radians = np.radians((360 / 365) * (DOY - 81))

        E = 9.87 * np.sin(2 * B_radians) - 7.53 * np.cos(B_radians) - 1.5 * np.sin(B_radians)

        return E
    except Exception as e:
        return e


def solar_time(std_datetime_GMT):
    """This function computes solar time using
    GMT stardard time and prime green which meridan as a reference
    returns: solar time in hours"""

    try:
        global longitude
        equation_of_time = equation_of_time_pvcdrom(std_datetime_GMT.tz_convert("Africa/Kampala"))
        tm = std_datetime_GMT
        tm_hr = tm.hour + tm.minute / 60

        solartime = tm_hr + (equation_of_time + 4 * longitude) / 60

        return solartime
    except Exception as e:
        return e


def hour_angle(sdt_datetime_GMT):
    """Hour angle and zero at solar noon
    std_GMT_time: str
    local standard time  in GMT

    hour_anlge:numeric
       hour_angle in degrees"""

    try:
        sol_time = solar_time(sdt_datetime_GMT)

        hr_angle = 15 * (sol_time - 12)

        return hr_angle
    except Exception as e:
        return e


def angle_of_elevation(sdt_datetime_GMT):
    """This function calculates angle of incidence
    std_GMT_time:str
      standard time in GMT
    angle_of_elevation:numeric
     angle of incidence in degrees"""

    try:
        global latitude
        hr_angle = hour_angle(sdt_datetime_GMT)
        decli = declination_cooper69(sdt_datetime_GMT)

        aoe = np.arcsin(np.sin(np.radians(decli)) * np.sin(np.radians(latitude)) + np.cos(np.radians(decli)) * np.cos(
            np.radians(latitude)) * np.cos(np.radians(hr_angle)))

        aoe_deg = np.degrees(aoe)
        return aoe_deg

    except Exception as e:
        return e


def solar_azimuth(sdt_datetime_GMT):
    """This function calculates solar azimuth
      std_GMT_time:str
        standard time in GMT
      angle of azimuth:numeric
        angle of azimuth in degrees"""
    try:
        global latitude
        ofe = angle_of_elevation(sdt_datetime_GMT)
        decli = declination_cooper69(sdt_datetime_GMT)
        num = np.sin(np.radians(ofe)) * np.sin(np.radians(latitude)) - np.sin(np.radians(decli))
        den = np.cos(np.radians(ofe)) * np.cos(np.radians(latitude))

        sol_az = np.arccos(num / den)
        sol_az_deg = np.degrees(sol_az)
        return sol_az_deg
    except Exception as e:
        return e


def angle_of_incidence(sdt_datetime_GMT):
    """This function calculates angle of incidence
     basing on stadard time GMT and computes angle of incidence """
    try:
        global surtilt
        global surazim
        hourangle = hour_angle(sdt_datetime_GMT)
        declination = declination_cooper69(sdt_datetime_GMT)
        aoi = np.arccos(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) * np.cos(np.radians(surtilt)) - np.sin(
                np.radians(declination)) * np.cos(np.radians(latitude)) * np.sin(np.radians(surtilt)) * np.cos(
                np.radians(surazim)) + np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * np.cos(
                np.radians(surtilt)) * np.cos(np.radians(hourangle)) + np.cos(np.radians(declination)) * np.sin(
                np.radians(latitude)) * np.sin(np.radians(surtilt)) * np.cos(np.radians(surazim)) * np.cos(
                np.radians(hourangle)) + np.cos(np.radians(declination)) * np.sin(np.radians(surtilt)) * np.sin(
                np.radians(hourangle)) * np.sin(np.radians(surazim)))
        aoi_deg = np.degrees(aoi)
        return aoi_deg
    except Exception as e:
        return e


def extra_irr_h(std_datetime_GMT):
    """This function calculates extraterrestrial irradiance
    on horizontal plane using different methods like spencer, asce and duffie_beckman models
    DOY:numeric
     Day of the year
    method:str
     "spencer","asce","duffie_and_beckman"
    Solar constant:numeric
     default 1367
    Angle_of_elev:numereic
     angle ofelevation in degrees
     retruns: array
     """

    try:
        global sc
        DOY = to_doy(std_datetime_GMT)
        B_radian = np.radians(day_angle(std_datetime_GMT))
        aoe_radians = np.radians(angle_of_elevation(std_datetime_GMT))

        E_spencer = 1.00011 + 0.034221 * np.cos(B_radian) + 0.00128 * np.sin(B_radian) + 0.000719 * np.cos(
            2 * B_radian) + 0.000077 * np.sin(2 * B_radian)

        E_asce = 1 + 0.033 * np.cos(B_radian)

        E_duffie_and_beckman = 1 + 0.033 * np.cos(np.radians((360 / 365) * DOY))

        return np.array([x * np.sin(aoe_radians) * sc for x in [E_spencer, E_asce, E_duffie_and_beckman]])
    except Exception as e:
        return e


def extra_irr(std_datetime_GMT):
    """This function calculates extraterrestrial irradiance
    using different methods
    DOY:numeric
     Day of the year
    method:str
     "spencer","asce","duffie_and_beckman"
    Solar constant:numeric
     default 1367
    Angle_of_elev:numereic
     angle ofelevation in degrees"""

    try:
        global sc
        DOY = to_doy(std_datetime_GMT)
        B_radian = np.radians(day_angle(std_datetime_GMT))
        aoe_radians = np.radians(angle_of_elevation(std_datetime_GMT))

        E_spencer = 1.00011 + 0.034221 * np.cos(B_radian) + 0.00128 * np.sin(B_radian) + 0.000719 * np.cos(
            2 * B_radian) + 0.000077 * np.sin(2 * B_radian)

        E_asce = 1 + 0.033 * np.cos(B_radian)

        E_duffie_and_beckman = 1 + 0.033 * np.cos(np.radians((360 / 365) * DOY))

        return np.array([x * sc for x in [E_spencer, E_asce, E_duffie_and_beckman]])
    except Exception as e:
        return e


def erbs(std_GMT_time, GHI):
    """this function calculates beam horizontal irradiance
    and diffuse horrizontal iirradiace for three clear sky models using
    erbs model"""

    Extr = std_GMT_time.apply(extra_irr_h)
    kt = GHI / Extr
    kt = np.stack(kt.values)
    kt = np.maximum(kt, 0)
    phi = np.where(kt <= 0.22, 1 - 0.09 * kt, np.where((kt > 0.22) & (kt <= 0.8),
                                                       0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                                                       0.165))
    GHI = GHI[:, np.newaxis]
    BHI = GHI * (1 - phi)
    DHI = GHI - BHI
    return BHI


def orgil_hollands(std_GMT_time, GHI):
    """this function calculates beam horizontal irradaince and diffuse irradaince
    using orgill and hollands model using clear sky models as input"""
    Extr = std_GMT_time.apply(extra_irr_h)
    kt = GHI / Extr
    kt = np.stack(kt.values)
    kt = np.maximum(kt, 0)
    phi = np.where(kt < 0.35, 1 - 0.249 * kt, np.where((kt >= 0.35) & (kt <= 0.75), 1.577 - 1.84 * kt, 0.177))
    GHI = GHI[:, np.newaxis]
    BHI = GHI * (1 - phi)
    DHI = GHI - BHI
    return BHI


def bolands(std_GMT_time, GHI):
    """this function evalutes beam horizontal irradaince and diffuse irradiance
    """
    Extr = std_GMT_time.apply(extra_irr_h)
    kt = GHI / Extr
    kt = np.stack(kt.values)
    kt = np.where(kt < 0, 0.587, kt)
    den = 1 + np.exp(7.997 * (kt - 0.587))
    phi = 1 / den
    GHI = GHI[:, np.newaxis]
    BHI = GHI * (1 - phi)
    DHI = GHI - BHI
    return BHI


# this is the model i developed using local datasets without
# partitioning it

def noor_diffuse4(std_GMT_time, GHI):
    Extr = std_GMT_time.apply(extra_irr_h)
    kt = GHI / Extr
    kt = np.stack(kt.values)
    kt = np.maximum(kt, 0)
    phi = np.where((kt >= 0) & (kt <= 1),
                   -6.40971573 * kt ** 4 + 17.32080925 * kt ** 3 + -15.29575829 * kt ** 2 + 4.53944271 * kt + 0.252237,
                   np.nan)
    GHI = GHI[:, np.newaxis]
    BHI = GHI * (1 - phi)
    return BHI


# this is the model i developed using local datasets but partioned

def noor_diffuse(std_GMT_time, GHI):
    Extr = std_GMT_time.apply(extra_irr_h)
    kt = GHI / Extr
    kt = np.stack(kt.values)
    kt = np.maximum(kt, 0)
    phi = np.where(kt < 0.2,
                   0.09253534928937057 + 3.27719669e+01 * kt + -6.69912075e+02 * kt ** 2 + 6.57869326e+03 * kt ** 3 + -3.00603914e+04 * kt ** 4 + 5.15852035e+04 * kt ** 5,
                   np.where((kt >= 0.2) & (kt < 0.8),
                            0.57780607 + 1.30748683 * kt + -4.75465657 * kt ** 2 + 3.43032779 * kt ** 3,
                            np.where((kt >= 0.8) & (kt <= 1),
                                     0.8532920137087421 + -1.64520004 * kt + 1.22166737 * kt ** 2, np.nan)))
    GHI = GHI[:, np.newaxis]
    BHI = GHI * (1 - phi)
    return BHI


def poa_b(std_GMT_time, BHI):
    """this function calculates direct  irradiance on
     inclined plane"""
    aoi = np.stack(std_GMT_time.apply(angle_of_incidence).values)
    aoi_radians = np.radians(aoi)
    aoe_radians = np.radians(std_GMT_time.apply(angle_of_elevation))
    if aoi_radians.ndim == 2:
        aoe_radians = aoe_radians[:, np.newaxis]
    POA_b = BHI * (np.cos(aoi_radians) / np.sin(aoe_radians))
    return np.maximum(POA_b, 0)


def lie_jordan(DHI):
    """this function calculates diffuse irradiance on
    tilt collector from diffuse horrizontal irradiance"""
    global surtilt
    poa_d = 0.5 * DHI * (1 + np.cos(np.radians(surtilt)))
    return poa_d


def ground_reflection(ghi):
    """this function computes irradiance from reflected irradiance"""
    global surtilt
    albedo = 0.2
    if type(ghi) == np.ndarray and not type(surtilt) == int:
        if surtilt.ndim == 2:
            reflect_irrad = [GHI * albedo * (1 - np.cos(np.radians(surtilt))) * 0.5 for GHI in ghi.ravel()]
            return reflect_irrad
    reflect_irrad = ghi * albedo * (1 - np.cos(np.radians(surtilt))) * 0.5
    return reflect_irrad


def hay_davies(std_GMT_time,BHI, GHI,G_ext):
    """This function  calculates beam horinzontal
    irradiance using hay and davies"""
    global surtilt
    try:
        DHI = GHI - BHI
        f_hay = BHI/G_ext
        f_hay = f_hay.fillna(0)  # filling na as result of 0/0
        aoi_radians = np.radians(std_GMT_time.apply(angle_of_incidence))
        aoe_radians = np.radians(std_GMT_time.apply(angle_of_elevation))

        poa_d = DHI*(f_hay*(np.cos(aoi_radians)/np.sin(aoe_radians)) + (0.5*(1+np.cos(surtilt)))*(1-f_hay))
        return np.maximum(poa_d,0)
    except ZeroDivisionError:
        return 0


def perez(std_GMT_time, BHI, GHI, G_ext):
    """this function calaculates diffuse componet
    on tilted collector"""
    global surtilt
    try:
        DHI = GHI - BHI
        arr = np.array([[-0.008, 0.588, -0.062, -0.060, 0.072, -0.022], [0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
                        [0.330, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                        [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
                        [1.060, -1.600, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.250, 0.156, -1.377, 0.251]])
        labels = pd.cut([1.060, 1.20, 1.4, 1.8, 2.5, 3.0, 5, 7],
                        [1.000, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200, np.inf], right=False)
        df = pd.DataFrame(arr, index=labels.categories, columns=["F11", "F12", "F13", "F21", "F22", "F23"])

        aoe_radians = np.radians(std_GMT_time.apply(angle_of_elevation))
        zenith_radians = np.pi - aoe_radians

        aoi = np.stack(std_GMT_time.apply(angle_of_incidence).values)
        aoi_radians = np.radians(aoi)

        air_mass = 1 / np.sin(aoe_radians)
        G_ext = G_ext / np.sin(aoe_radians)
        delta = air_mass * DHI / G_ext
        delta = delta.fillna(0)  # nan generated from 0/0
        dni = BHI / np.sin(aoe_radians)
        eps = ((DHI + dni) / DHI + 1.041 * zenith_radians ** 3) / (1 + 1.041 * zenith_radians ** 3)
        eps = eps.fillna(0)  # nan generated from 0/0
        coeficients = []
        for value in eps.values:
            if value == 0:
                coeficients.append([0] * 6)
            else:
                for x, y in enumerate(df.index):
                    if value in y:
                        coef = df.iloc[x, :]
                        coeficients.append(coef)
                        break
                else:
                    coeficients.append([np.nan] * 6)
        coef = np.stack(coeficients)
        F1 = np.maximum(0, (coef[:, 0] + coef[:, 1] * delta + coef[:, 2] * zenith_radians))
        F2 = coef[:, 3] + coef[:, 4] * delta + zenith_radians * coef[:, 5]

        a = np.maximum(0, np.cos(aoi_radians))
        b = np.maximum(np.cos(np.radians(85)), np.cos(zenith_radians))
        if a.ndim == 2:
            F1 = F1[:, np.newaxis]
            F2 = F2[:, np.newaxis]
            b = b[:, np.newaxis]
            DHI = DHI[:, np.newaxis]
        poa_d = DHI * (0.5 * (1 + np.cos(np.radians(surtilt))) * (1 - F1) + F1 * (a / b) + F2 * np.sin(
            np.radians(surtilt)))
        return np.maximum(poa_d, 0)
    except ZeroDivisionError:
        return 0


def rmbd(measured_poa, modeled_poa):
    """This function calaculates Relative mean bias deviation """
    evalute = pd.DataFrame({"measured_poa": measured_poa.values, "modeled_poa": modeled_poa.values})

    evalute = evalute[~evalute.isin([np.inf, np.nan]).any(1)]
    evalute = evalute[evalute.measured_poa > 0]
    rela_mbd = (evalute.measured_poa - evalute.modeled_poa).mean() / evalute.measured_poa.mean()

    return rela_mbd * 100


def rmsd(measured_poa, modeled_poa):
    """This function calculates root mean square deviation"""
    evalute = pd.DataFrame({"measured_poa": measured_poa.values, "modeled_poa": modeled_poa.values},
                           index=measured_poa.index)

    evalute = evalute[~evalute.isin([np.inf, np.nan]).any(1)]
    evalute = evalute[evalute.measured_poa > 0]
    error = np.sqrt(((evalute.measured_poa - evalute.modeled_poa) ** 2).mean()) / evalute.measured_poa.mean()
    return error * 100

