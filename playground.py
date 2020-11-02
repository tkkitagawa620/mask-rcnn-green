# def xml2csv(xml_path, image_path):
#     """Convert XML to CSV
#
#     Args:
#         xml_path (str): Location of annotated XML file
#     Returns:
#         pd.DataFrame: converted csv file
#
#     """
#     print("xml to csv {}".format(xml_path))
#     xml_list = []
#     xml_df = pd.DataFrame()
#
#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         h, w = img.shape[:2]
#
#         for member in root.findall('object'):
#             dimensions = exrtract_dimensions(member.find('polygon'))
#             value = (ommit_escape(root.find('filename').text),
#                     # int(root.find('size')[1].text),
#                      int(w),
#                      int(h),
#                      ommit_escape(member[0].text),
#                      dimensions[0],
#                      dimensions[1],
#                      dimensions[2],
#                      dimensions[3]
#                      )
#             xml_list.append(value)
#             print(value)
#             column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#             xml_df = pd.DataFrame(xml_list, columns=column_name)
#     except Exception as e:
#         print('xml conversion failed:{}'.format(e))
#         return pd.DataFrame(columns=['filename,width,height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
#     return xml_df


# def df2labelme(symbolDict, image_path, image):
#     """ convert annotation in CSV format to labelme JSON
#
#     Args:
#         symbolDict (dataframe): annotations in dataframe
#         image_path (str): path to image
#         image (np.ndarray): image read as numpy array
#
#     Returns:
#         JSON: converted labelme JSON
#
#     """
#     try:
#         symbolDict['min'] = symbolDict[['xmin', 'ymin']].values.tolist()
#         symbolDict['max'] = symbolDict[['xmax', 'ymax']].values.tolist()
#         symbolDict['points'] = symbolDict[['min', 'max']].values.tolist()
#         symbolDict['shape_type'] = 'rectangle'
#         symbolDict['group_id'] = None
#         height, width, _ = image.shape
#         symbolDict['height'] = height
#         symbolDict['width'] = width
#         encoded = base64.b64encode(open(image_path, "rb").read())
#         symbolDict.loc[:, 'imageData'] = encoded
#         symbolDict.rename(
#             columns={'class': 'label', 'filename': 'imagePath', 'height': 'imageHeight', 'width': 'imageWidth'},
#             inplace=True)
#         converted_json = (symbolDict.groupby(['imagePath', 'imageWidth', 'imageHeight', 'imageData'], as_index=False)
#                           .apply(lambda x: x[['label', 'points', 'shape_type', 'group_id']].to_dict('r'))
#                           .reset_index()
#                           .rename(columns={0: 'shapes'})
#                           .to_json(orient='records'))
#         converted_json = json.loads(converted_json)[0]
#     except Exception as e:
#         converted_json = {}
#         print('error in labelme conversion:{}'.format(e))
#     return converted_json


# xml_path = "data_annotated/p1010572.xml"
# image_path = 'data_annotated/p1010572.jpg'
#
# xml_csv = xml2csv(xml_path, image_path)
# image = cv2.imread(image_path)
# csv_json = df2labelme(xml_csv, image_path, image)
#
# with open('data_dataset_coco/p1010572.json', 'w') as outfile:
#     json.dump(csv_json, outfile)
