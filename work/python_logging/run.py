from python_logging.first_class import FirstClass
from python_logging.second_class import SecondClass

number = FirstClass()
number.increment_number()
number.increment_number()
print ("Current number: %s" % str(number.current_number))
number.clear_number()
print ("Current number: %s" % str(number.current_number))

system = SecondClass()
system.enable_system()
system.disable_system()
print("Current system configuration: %s" % str(system.enabled))

# print a varaible value
number.contain_varaibles()
# Show 'capture exception with traceback
number.show_captue_exception()
