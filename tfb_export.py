import os
import scipy.misc
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


def save_images_from_event(fn, tag, output_dir='./'):
	assert(os.path.isdir(output_dir))

	image_str = tf.placeholder(tf.string)
	im_tf = tf.image.decode_image(image_str)

	sess = tf.InteractiveSession()
	with sess.as_default():
		count = 0
		for e in tf.train.summary_iterator(fn):
			for v in e.summary.value:
				# if v.tag == tag:
				im = im_tf.eval({image_str: v.image.encoded_image_string})
				output_fn = os.path.realpath('{}/{}.png'.format(output_dir, v.tag))
				print("Saving '{}'".format(output_fn))
				scipy.misc.imsave(output_fn, im)
				count += 1  

save_images_from_event('log/GAN_xs2xs_inter_0_05-05-22:52:26/logs/events.out.tfevents.1557130097.cpu5admin', None, 'vis_images')