'''Check the example pipeline .csv output against a reference

Usage:
python examplepipelinestatistics --test-name <test-name>
                                 --test-file <test-file> 
                                 --reference-file <reference-file>
                                 --output <output-xml-file>

The output is in the JUnit format:
<testsuite name="<test-name>" test-file="<file-name>" 
            reference-file="<file-name>" tests="#" failures="#" >
  <testcase name="<feature-name>" ref-mean="#" test-mean="#" ref-std="#" test-std="#" test-nan="#", ref-nan="#">
    <error type="<type>" message="message" />  
    <failure type="<type>" message="<message>" />
</testsuite>
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import csv
import numpy as np
import traceback

OUTPUT_PRESENT = "output-present"
MATCHING_COLUMNS = "matching-columns"
MEASUREMENT_COUNT = "measurement-count"

'''Error was found during logical analysis of data'''
ERROR_TYPE_LOGICAL = "logical"

'''Error was due to different quantities of data'''
ERROR_TYPE_QUANTITY = "quantity"

'''Error was due to differing values among data items'''
ERROR_TYPE_MEASUREMENT  = "measurement"


def test_files(test_name, test_file, reference_file, output_file,
               max_deviation = .1, max_nan_deviation = .1, 
               max_obj_deviation = .1):
    '''Compare a test file against a reference file, generating output xml
    
    test_name - the name that appears in the test-suite
    test_file - the .csv file containing the test data
    reference_file - the .csv file containing the reference data
    output_file - the .xml file that will have the test output
    max_deviation - the maximum deviation of the mean value in standard
                    deviation units
    max_nan_deviation - the maximum deviation in the # of nans relative to
                        the sample size
    max_obj_deviation - the maximum deviation in the # of objects per image set
    '''
    output_fd = open(output_file, "w")
    output_fd.write("""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="%(test_name)s"
            test-file="%(test_file)s"
            reference-file="%(reference_file)s"
            max-allowed-deviation="%(max_deviation)f"
            max-allowed-nan-deviation="%(max_nan_deviation)f"
            max-allowed-obj-deviation="%(max_obj_deviation)f"
""" % locals())
    
    try:
        test_reader = csv.reader(open(test_file, "r"))
        reference_reader = csv.reader(open(reference_file, "r"))
        test_measurements = collect_measurements(test_reader)
        reference_measurements = collect_measurements(reference_reader)
        statistics = test_matching_columns(test_measurements, reference_measurements)
        statistics += test_deviations(test_measurements, reference_measurements,
                                      max_deviation, max_nan_deviation, 
                                      max_obj_deviation)
        if (test_measurements.has_key("ObjectNumber") and
            reference_measurements.has_key("ObjectNumber")):
            image_numbers = np.unique(np.hstack((
                test_measurements["ImageNumber"], 
                reference_measurements["ImageNumber"])))
            test_measurements_per_image = collect_per_image(test_measurements,
                                                            image_numbers)
            reference_measurements_per_image = collect_per_image(
                reference_measurements, image_numbers)
            for feature in test_measurements_per_image.keys():
                test_measurement = test_measurements_per_image[feature]
                reference_measurement = reference_measurements_per_image[feature]
                fs_all = None
                for i, tm, rm in zip(image_numbers, test_measurement, reference_measurement):
                    fs = test_deviation(feature, tm, rm, 
                                        max_deviation, max_nan_deviation,
                                        max_obj_deviation, True)
                    if any(statistic[2] is not None for statistic in fs):
                        fs_all = fs
                        break
                    for s in fs:
                        s[1]["ImageNumber"] = i
                    if fs_all is None:
                        fs_all = [(name, [attributes], error_type, message, body)
                                  for name, attributes, error_type, message, body in fs]
                    else:
                        for i in range(len(fs)):
                            fs_all[i][1].append(fs[i][1])
                statistics += fs_all
    except Exception, e:
        stacktrace = traceback.format_exc()
        message = e.message
        output_fd.write("""            errors="0"
            failures="1">
  <testcase name="%s">
    <failure type="%s" message="%s">
%s
    </failure>
  </testcase>
</testsuite>
""" % ( OUTPUT_PRESENT, type(e), message, stacktrace))
        output_fd.close()
        return
    
    error_count = count_errors(statistics)
    output_fd.write("""            errors="%d"
            failures="0">
""" % error_count)
    for statistic in statistics:
        write_statistic(output_fd, statistic)
    output_fd.write("</testsuite>\n")
    output_fd.close()

ignore_categories = (
    "ExecutionTime",
    "ModuleError",
    "PathName")

def make_success_statistic(name, attributes, per_image = False):
    if per_image:
        name += "_per_image"
    return (name, attributes, None, None, None)

def make_error_statistic(name, attributes, error_type, message, body,
                         per_image = False):
    if per_image:
        name += "_per_image"
    return (name, attributes, error_type, message, body)

def write_statistic(output_fd, statistic):
    name, attributes, error_type, message, body = statistic
    output_fd.write('  <testcase name="%s" ' % name)
    if isinstance(attributes, list):
        output_fd.write(">\n")
        for attribute in attributes:
            output_fd.write("    <success %s />\n" %
                            " ".join(['%s="%s"' % (key, str(value))
                                      for key, value in attribute.iteritems()]))
        output_fd.write("  </testcase>\n")
    elif error_type is None:
        output_fd.write(' '.join(['%s="%s"' % (key, str(value))
                                  for key, value in attributes.iteritems()]))
        output_fd.write('/>\n')
    else:
        output_fd.write('>\n')
        output_fd.write('    <error type="%s" message="%s"\n>' %
                        (error_type, message))
        output_fd.write(body)
        output_fd.write('    </error>\n')
        output_fd.write('  </testcase>\n')
        
def count_errors(statistics):
    result = 0
    for name, attributes, error_type, message, body in statistics:
        if error_type is not None:
            result += 1
    return result

def collect_measurements(rdr):
    '''Create a dictionary of feature name to vector of measurements
    
    rdr - a csv reader
    '''
    header = rdr.next()
    d = {}
    for field in header:
        ignore = False
        for ignore_category in ignore_categories:
            if field.find(ignore_category) != -1:
                ignore = True
                break
        if ignore:
            continue
        d[field] = []
        
    for i,row in enumerate(rdr):
        if len(row) != len(header):
            raise ValueError("Row size (%d) doesn't match header size (%d) at line %d" %
                             (len(row), len(header), i+1))
        for value, field in zip(row, header):
            if d.has_key(field):
                d[field].append(value)
    #
    # Make Numpy arrays
    #
    for field in d.keys():
        d[field] = np.array(d[field])
    #
    # Try float casts
    #
    for field in d.keys():
        try:
            tmp = d[field]
            tmp_not_nan = tmp[tmp != 'nan'].astype(np.float32)
            if (np.all(tmp_not_nan == tmp_not_nan.astype(int)) and
                not np.any(tmp == 'nan')):
                tmp_out = np.zeros(len(tmp), int)
                tmp_not_nan = tmp_not_nan.astype(int)
            else:
                tmp_out = np.zeros(len(tmp), np.float32)
            if np.any(tmp == 'nan'):
                tmp_out[tmp == 'nan'] = np.nan
                tmp_out[tmp != 'nan'] = tmp_not_nan
            else:
                tmp_out = tmp_not_nan
            d[field] = tmp_out
        except:
            pass
    return d

def collect_per_image(measurements, image_numbers):
    image_indexes = measurements["ImageNumber"]
    result = {}
    for key in measurements.keys():
        result[key] = [measurements[key][image_indexes == i]
                       for i in image_numbers]
    return result
        
    
def test_matching_columns(test_measurements, reference_measurements):
    '''Ensure that the test and reference measurements have the same features
    
    Has side-effect of deleting measurements that are present in one
    but missing in other.
    '''
    assert isinstance(test_measurements, dict)
    assert isinstance(reference_measurements, dict)
    missing_in_test = []
    missing_in_reference = []
    for feature in test_measurements.keys():
        if not reference_measurements.has_key(feature):
            missing_in_reference.append(feature)
    for feature in reference_measurements.keys():
        if not test_measurements.has_key(feature):
            missing_in_test.append(feature)

    for feature in missing_in_test:
        del reference_measurements[feature]
    
    for feature in missing_in_reference:
        del test_measurements[feature]
    
    if len(missing_in_reference) + len(missing_in_test) > 0:
        body = ""
        if len(missing_in_reference):
            body += ("Measurements not present in reference:\n%s" %
                     '\n\t'.join(missing_in_reference))
            message = "Test measurements contain additional features"
        if len(missing_in_test):
            body += ("Measurements missing from test:\n%s" %
                     '\n\t'.join(missing_in_test))
            if len(missing_in_reference):
                message += " and test measurements are missing features"
            else:
                message = "Test measurements are missing features"
        return [make_error_statistic(MATCHING_COLUMNS, {}, 
                                     ERROR_TYPE_LOGICAL, message, body)]
    return []


def test_deviations(test_measurements, reference_measurements, 
                    max_deviation, max_nan_deviation, max_obj_deviation,
                    per_image = False):
    statistics = []
    feature = test_measurements.keys()[0]
    tm_len = len(test_measurements[feature])
    rm_len = len(reference_measurements[feature])
    if tm_len+rm_len > 0:
        deviance = (float(abs(tm_len - rm_len)) /
                    float(tm_len + rm_len))
        if deviance > max_obj_deviation:
            message = ("# of measurements is different: %d in test, %d in reference" %
                       (tm_len, rm_len))
            s = make_error_statistic(MEASUREMENT_COUNT, {},
                                     ERROR_TYPE_QUANTITY, message, "", 
                                     per_image)
            statistics += [s]
    for feature in test_measurements.keys():
        statistics += test_deviation(feature,
                                     test_measurements[feature],
                                     reference_measurements[feature],
                                     max_deviation,
                                     max_nan_deviation,
                                     max_obj_deviation,
                                     per_image)
    return statistics

def test_deviation(feature, test_measurement, reference_measurement,
                   max_deviation, max_nan_deviation, max_obj_deviation,
                   per_image):
    statistics = []
    if test_measurement.dtype == np.float32:
        return test_float_deviation(feature, test_measurement, 
                                    reference_measurement,
                                    max_deviation, max_nan_deviation,
                                    per_image)
    elif test_measurement.dtype == int:
        return test_integer_deviation(feature, test_measurement,
                                      reference_measurement,
                                      max_deviation, per_image)
    else:
        return test_string_deviation(feature, test_measurement,
                                     reference_measurement, per_image)

def test_float_deviation(feature, test_measurement, reference_measurement,
                         max_deviation, max_nan_deviation, per_image):
    tm_no_nan = test_measurement[~ np.isnan(test_measurement )]
    rm_no_nan = reference_measurement[~ np.isnan(reference_measurement)]
    tm_nan_fraction = 1.0 - float(len(tm_no_nan)) / float(len(test_measurement))
    rm_nan_fraction = 1.0 - float(len(rm_no_nan)) / float(len(reference_measurement))
    if tm_nan_fraction + rm_nan_fraction > 0:
        nan_deviation = (abs(tm_nan_fraction - rm_nan_fraction) /
                         (tm_nan_fraction + rm_nan_fraction))
        if nan_deviation > max_nan_deviation:
            message = ("# of NaNs differ: %d in test vs %d in reference" %
                       (np.sum(np.isnan(test_measurement)),
                        np.sum(np.isnan(reference_measurement))))
            s = make_error_statistic(feature, {}, ERROR_TYPE_QUANTITY,
                                     message, "", per_image)
            return [s]
    test_mean = np.mean(tm_no_nan) 
    reference_mean = np.mean(rm_no_nan)

    sd = (np.std(tm_no_nan) + np.std(rm_no_nan)) / 2.0
    sd = max(sd, .000001, .00001 * (test_mean+reference_mean) / 2.0)
    mean_diff = abs(test_mean - reference_mean) / sd
    if mean_diff > max_deviation:
        message = ("Test and reference means differ: %f / %f test, %f / %f reference" %
                   (test_mean, np.std(tm_no_nan), reference_mean,
                    np.std(rm_no_nan)))
        s = make_error_statistic(feature, {}, ERROR_TYPE_MEASUREMENT,
                                 message, "", per_image)
        return [s]
    
    attributes = dict(test_mean = test_mean,
                      reference_mean = reference_mean,
                      sd = sd,
                      test_nan_fraction = tm_nan_fraction,
                      reference_nan_fraction = rm_nan_fraction)
    return [make_success_statistic(feature, attributes, per_image)]

def test_integer_deviation(feature, test_measurement, reference_measurement,
                           max_deviation, per_image):
    do_like_float = False
    for allowed_feature in ("count", "area"):
        if feature.lower().find(allowed_feature) != -1:
            do_like_float = True
            break
    if do_like_float:
        return test_float_deviation(feature, 
                                    test_measurement.astype(np.float32),
                                    reference_measurement.astype(np.float32),
                                    max_deviation, 1, per_image)
    return []

def test_string_deviation(feature, test_measurement, reference_measurement,
                          per_image):
    if len(test_measurement) != len(reference_measurement):
        return []
    
    indexes = np.argwhere(test_measurement != reference_measurement)
    if len(indexes != 0):
        body = '\n'.join(
            ["%d: t=%s, r=%s" % 
             (i+1, test_measurement[i], reference_measurement[i])
             for i in indexes])
        message = "text measurements differ"
        return [make_error_statistic(feature, {}, ERROR_TYPE_MEASUREMENT,
                                     message, body, per_image)]
    return [ make_success_statistic(feature, {}, per_image) ]

if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option("-n", "--test-name",
                      dest="test_name",
                      default="PipelineTest",
                      help="The name of the test suite")
    parser.add_option("-t", "--test-file",
                      dest = "test_file",
                      default = "test.csv",
                      help="The path to the file containing the test data")
    parser.add_option("-r", "--reference-file",
                      dest = "reference_file",
                      default = "reference.csv",
                      help="The path to the file containing the reference data")
    parser.add_option("-o", "--output-file",
                      dest = "output_file",
                      default = "out.xml",
                      help="The path to the file to contain the JUnit-style test results")

    options, args = parser.parse_args()
    test_files(options.test_name,
               options.test_file,
               options.reference_file,
               options.output_file)
