import mujoco
# 加载 URDF
model = mujoco.MjModel.from_xml_path("/data2/yangky/test/GMR/assets/t800/serial_t800.urdf")
# 保存为 XML
mujoco.mj_saveLastXML("/data2/yangky/test/GMR/assets/t800/t800.xml", model)
print("Conversion done!")