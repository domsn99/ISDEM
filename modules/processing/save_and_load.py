import re
from pathlib import Path

def all_e_files(url, reg_ex ='**/data/*.npz'):
    ''' finds, sorts and lists all files in url, according to naming convention '**/electrons/*.npy'
    '''
    e_files = list(url.glob(reg_ex))
    dummy = [str(f).split('\\')[-1].lower() for f in e_files]   # Convert to lower case
    dummy = [float(f.split('-')[0]) for f in dummy]   # Convert to lower case
    dummy = np.argsort(np.array(dummy))
    sorted = [e_files[i] for i in dummy]
    print('{} electron packages found:'.format(len(sorted)))
    #for i in sorted: print(i)
    print('-------------------')
    return sorted 

def find_subdirs_with_matching_files(root_dir, reg_ex = 'data.*\.npz'):
    '''
    Finds all subdirectories that contain files with:
    RelativePath of file to subdirectory matchen reg_ex
    '''
    matching_subdirs = set()
    pattern = re.compile(reg_ex)
    
    # Walk through root_dir
    for parent in Path(root_dir).rglob('*'):
        for decendent in Path(parent).rglob('*'):
            if pattern.match(str(decendent.relative_to(parent))):
                matching_subdirs.add(parent)
    return list(matching_subdirs)

def build_session_dict(root_path, measurements, para_reg_eg = None, sub_reg_ex = 'data.*\.npz'):
    Session = {}
    Session['measurements'] = {}
    for measurement in measurements:
        Session['measurements'][measurement] = {}
        Session['measurements'][measurement]['submeasurements'] = {}
        measurement_path = root_path / measurement
        Session['measurements'][measurement]['path'] = measurement_path
        submeasurements = find_subdirs_with_matching_files(measurement_path)
        for submeasurement in submeasurements:
            Session['measurements'][measurement]['submeasurements'][submeasurement.name] = {}
            Session['measurements'][measurement]['submeasurements'][submeasurement.name]['files'] = list((submeasurement).rglob('*data/*.npz'))
            Session['measurements'][measurement]['submeasurements'][submeasurement.name]['path'] = submeasurement
    return Session 