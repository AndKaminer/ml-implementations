terraform {
  backend "s3" {
    bucket  = "lightweight-ml-serving-api-terraform-state-bucket"
    key     = "envs/dev/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "app_bucket" {
  bucket = "${var.app_name}-deployment-bucket"
  force_destroy = true
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_object" "app_zip" {
  bucket = aws_s3_bucket.app_bucket.id
  key    = "app.zip"
  source = var.app_zip_path
  etag   = filemd5(var.app_zip_path)
}

resource "aws_elastic_beanstalk_application" "app" {
  name = var.app_name
}

resource "aws_elastic_beanstalk_application_version" "version" {
  name        = "${var.app_name}-v${timestamp()}"
  application = aws_elastic_beanstalk_application.app.name
  bucket      = aws_s3_object.app_zip.bucket
  key         = aws_s3_object.app_zip.key
}

resource "aws_elastic_beanstalk_environment" "env" {
  name                = "${var.app_name}-env"
  application         = aws_elastic_beanstalk_application.app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.6.1 running Docker"
  version_label       = aws_elastic_beanstalk_application_version.version.name
  cname_prefix        = "${var.cname_prefix}"

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t3.micro"
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance"
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.eb_instance_profile.name
  }
}

resource "aws_iam_role" "eb_instance_role" {
  name = "${var.app_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eb_ec2_policy" {
  role       = aws_iam_role.eb_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
}

resource "aws_iam_instance_profile" "eb_instance_profile" {
  name = "${var.app_name}-instance-profile"
  role = aws_iam_role.eb_instance_role.name
}
